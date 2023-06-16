#include <libavcodec/avcodec.h>

#include <libavformat/avformat.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>

#include <pthread.h>
#include <unistd.h>

#include "libswscale/swscale.h"
#include "libavutil/imgutils.h"

//export function define
extern int init_decoder();
extern void close_decoder();
extern int decode_frame(char* pktdata, int pktsize, int width, int height, char* outdata);

#define USE_HW_CODEC 0
#define OUT_W 640
#define OUT_H 360

AVCodec *decoder = NULL;
static enum AVPixelFormat hw_pix_fmt;
enum AVHWDeviceType device_type;

uint8_t *bgrbuffer = NULL;
struct SwsContext* swsContext = NULL;
AVCodecContext *decoder_ctx = NULL;
static AVBufferRef *hw_device_ctx = NULL;

//encoder

int savepicture(unsigned char *colorimg, int width, int height);

static int hw_decoder_init(AVCodecContext *ctx, const enum AVHWDeviceType type)
{
    int err = 0;
    if ((err = av_hwdevice_ctx_create(&hw_device_ctx, type,
                                      NULL, NULL, 0)) < 0) {
        fprintf(stderr, "Failed to create specified HW device.\n");
        return err;
    }
    ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

    return err;
}

static enum AVPixelFormat get_hw_format(AVCodecContext *ctx,
                                        const enum AVPixelFormat *pix_fmts)
{
    const enum AVPixelFormat *p;
    for (p = pix_fmts; *p != -1; p++) {
        if (*p == hw_pix_fmt)
            return *p;
    }
    fprintf(stderr, "Failed to get HW surface format.\n");
    return AV_PIX_FMT_NONE;
}


int init_decoder()
{
    int res = -1;    
#if USE_HW_CODEC
    int i;
    device_type = AV_HWDEVICE_TYPE_CUDA;
    /* find the video decoder */
    decoder = avcodec_find_decoder_by_name("h264_cuvid");
    if (!decoder) {
        fprintf(stderr, "Codec not found\n");
        return res;
    }

    for (i = 0;; i++) {
        const AVCodecHWConfig *config = avcodec_get_hw_config(decoder, i);
        if (!config) {
            fprintf(stderr, "Decoder %s does not support device type %s.\n",
                    decoder->name, av_hwdevice_get_type_name(device_type));
            return res;
        }
        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
            config->device_type == device_type) {
            hw_pix_fmt = config->pix_fmt;
            break;
        }
    }

    decoder_ctx = avcodec_alloc_context3(decoder);
    if (!decoder_ctx) {
        fprintf(stderr, "Could not allocate codec context\n");
        return res;
    }

    // First set the hw device then set the hw frame
    decoder_ctx->get_format  = get_hw_format;

    int err = 0;
    if ((err = av_hwdevice_ctx_create(&hw_device_ctx, device_type,
                                      NULL, NULL, 0)) < 0) {
        fprintf(stderr, "Failed to create specified HW device.\n");
        return res;
    }
    decoder_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
#else
    decoder = avcodec_find_decoder_by_name("h264");
    if (!decoder) {
        fprintf(stderr, "Codec not found\n");
        return res;
    }
    decoder_ctx = avcodec_alloc_context3(decoder);
    if (!decoder_ctx) {
        fprintf(stderr, "Could not allocate codec context\n");
        return res;
    }
#endif
    /* open it */
    if (avcodec_open2(decoder_ctx, decoder, NULL) < 0) {
        fprintf(stderr, "Could not open codec\n");
        return res;
    }
    decoder_ctx->width = 1280;
    decoder_ctx->height = 720;
    res = 0;
    printf("video codec initialized.\n");
    return res;
}
void close_decoder()
{
    if(bgrbuffer) {
        av_freep(&bgrbuffer);
        bgrbuffer = NULL;
    }
    if (swsContext) {
        sws_freeContext(swsContext);
        swsContext = NULL;
    }
    if (decoder_ctx) {
        avcodec_free_context(&decoder_ctx);
        decoder_ctx = NULL;
    }
    if(hw_device_ctx) {
        av_buffer_unref(&hw_device_ctx);
        hw_device_ctx = NULL;
    }
    
}

int decode_frame(char* pktdata, int pktsize, int width, int height, char* outdata)
{
    AVFrame *decoded_frame = NULL;  
    AVPacket packet;
    AVFrame *pFrameBGR = NULL, *swFrame = NULL;
    int ret, numBytes = 0;
    int  res = -1;

    if (decoder_ctx == NULL) {
        return res;
    }    
    decoder_ctx->width = width;
    decoder_ctx->height = height;        

    decoded_frame = av_frame_alloc();
    if (decoded_frame == NULL) {
        return res;
    }
    av_init_packet(&packet);
    packet.data = pktdata;
    packet.size = pktsize;

    pFrameBGR = av_frame_alloc();
    swFrame = av_frame_alloc();
    if (NULL == pFrameBGR || NULL == swFrame) {
        goto fail;
    }
    if (bgrbuffer == NULL){
        numBytes = av_image_get_buffer_size(AV_PIX_FMT_BGR24, OUT_W, OUT_H, 1);        
        bgrbuffer = (uint8_t *)av_malloc(numBytes * sizeof(uint8_t));        
    }
    av_image_fill_arrays(pFrameBGR->data, pFrameBGR->linesize, bgrbuffer, AV_PIX_FMT_BGR24, OUT_W, OUT_H, 1);
    
    ret = avcodec_send_packet(decoder_ctx, &packet);
    if (ret < 0) {
        printf("avcodec_send_packet error.\n");
        goto fail;
    }    
    while (ret >= 0) {
        ret = avcodec_receive_frame(decoder_ctx, decoded_frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            printf("decode_frame need more avpacket.\n");
            goto fail;
        }
        else if (ret < 0) {
            fprintf(stderr, "Error during decoding\n");
            goto fail;
        }

#if USE_HW_CODEC
        // download frame from gpu to cpu
        ret = av_hwframe_transfer_data(swFrame, decoded_frame, 0);
        if (ret < 0) {
            fprintf(stderr, "Error transferring the data to system memory\n");
            goto fail;
        }
        if (swsContext == NULL) {
            swsContext = sws_getContext(decoder_ctx->width, decoder_ctx->height, AV_PIX_FMT_NV12, 
                                        OUT_W, OUT_H, AV_PIX_FMT_BGR24, SWS_BICUBIC, NULL, NULL, NULL);
        }
        sws_scale(swsContext, (const unsigned char* const*)swFrame->data, swFrame->linesize, 0, decoder_ctx->height, pFrameBGR->data, pFrameBGR->linesize);
#else
        if (swsContext == NULL) {
            swsContext = sws_getContext(decoder_ctx->width, decoder_ctx->height, AV_PIX_FMT_YUVJ420P, 
                                        OUT_W, OUT_H, AV_PIX_FMT_BGR24, SWS_BICUBIC, NULL, NULL, NULL);
            if(swsContext == NULL) {
                fprintf(stderr, "Can not open the scale context\n");
                goto fail;
            }
        }
        sws_scale(swsContext, (const unsigned char* const*)decoded_frame->data, decoded_frame->linesize, 0, decoder_ctx->height, pFrameBGR->data, pFrameBGR->linesize);
#endif        
        pFrameBGR->format = AV_PIX_FMT_BGR24;
        pFrameBGR->width = OUT_W;
        pFrameBGR->height = OUT_H;

        memcpy(outdata, bgrbuffer, pFrameBGR->width * pFrameBGR->height * 3);
        //test
        //savepicture(bgrbuffer, pFrameBGR->width, pFrameBGR->height);
        break;
    }
    res = 0;
fail:
    if(pFrameBGR)
        av_frame_free(&pFrameBGR);    
    if(swFrame)
        av_frame_free(&swFrame);    
    if(decoded_frame)   
        av_frame_free(&decoded_frame);
    
    av_packet_unref(&packet);
    return res;
}

// test code
#if 0
int framnum = 0;
#define UCHAR unsigned char
#define WORD unsigned short
#define UINT unsigned int
#define BOOL unsigned char
#define Uint unsigned int
#define BYTE unsigned char
#define DWORD unsigned int
#pragma pack(1)
typedef struct tagBITMAPFILEHEADER {
	WORD    bfType;
	DWORD   bfSize;
	WORD    bfReserved1;
	WORD    bfReserved2;
	DWORD   bfOffBits;
} BITMAPFILEHEADER, *PBITMAPFILEHEADER;

typedef struct tagBITMAPINFOHEADER{
	DWORD      biSize;
	long       biWidth;
	long       biHeight;
	WORD       biPlanes;
	WORD       biBitCount;
	DWORD      biCompression;
	DWORD      biSizeImage;
	long       biXPelsPerMeter;
	long       biYPelsPerMeter;
	DWORD      biClrUsed;
	DWORD      biClrImportant;
} BITMAPINFOHEADER, *PBITMAPINFOHEADER;
typedef struct tagRGBQUAD {
	BYTE    rgbBlue;
	BYTE    rgbGreen;
	BYTE    rgbRed;
	BYTE    rgbReserved;
} RGBQUAD;
#pragma pack()
int savepicture(unsigned char *colorimg, int width, int height) {

    char out_name[32];

    // Open file
    framnum ++;
    sprintf(out_name, "frame%d.bmp", framnum);

    	int i;
	int rwbytes;
	int delta;
	unsigned char temp[16], *pImage;
	BITMAPFILEHEADER bmpfh;
	BITMAPINFOHEADER bmpih;
	FILE* bmpFile;
#ifdef _TO_REAL_DRIVE_
	char buf[256];
	memset(buf, 0x00, sizeof(buf));
	strcpy(buf, bmpFileName);
	buf[0] = _TO_REAL_DRIVE_;
	bmpFileName = buf;
#endif

//return 0;
	bmpFile = fopen(out_name,"wb");
	if (bmpFile==NULL) return -1;

	bmpfh.bfType = 0x424D; //"BM";
	bmpfh.bfSize = sizeof(BITMAPFILEHEADER);
	bmpfh.bfReserved1 = 0;
	bmpfh.bfReserved2 = 0;
	bmpfh.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

	rwbytes = (int)((width*24+31)&(~31))/8;
	bmpih.biSize = sizeof(BITMAPINFOHEADER);
	bmpih.biWidth = width;
	bmpih.biHeight = height;
	bmpih.biPlanes = 1;
	bmpih.biBitCount = 24;
	bmpih.biCompression = 0;
	bmpih.biSizeImage = rwbytes*height;
	bmpih.biXPelsPerMeter = 0;
	bmpih.biYPelsPerMeter = 0;
	bmpih.biClrUsed = 0;
	bmpih.biClrImportant = 0;

	fwrite(&bmpfh,sizeof(BITMAPFILEHEADER),1,bmpFile);
	fwrite(&bmpih,sizeof(BITMAPINFOHEADER),1,bmpFile);

	pImage = colorimg + width*(height-1)*3;

	delta = rwbytes - width*3;
	for (i=0; i<height; i++)
	{
		//bmpFile.Write(pImage, width);
		fwrite(pImage,width*3,1,bmpFile);
		if (delta>0)
		{
			//bmpFile.Write(temp, delta);
			fwrite(temp,delta,1,bmpFile);
		}
		pImage -= (width*3);
	}

	//	bmpFile.Close();
	fclose(bmpFile);

    return 0;
}
#endif
