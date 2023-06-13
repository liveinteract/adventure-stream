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
AVCodec *decoder = NULL;
static enum AVPixelFormat hw_pix_fmt;
enum AVHWDeviceType device_type;

uint8_t *bgrbuffer = NULL;
struct SwsContext* swsContext = NULL;
AVCodecContext *decoder_ctx = NULL;
static AVBufferRef *hw_device_ctx = NULL;

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
    AVFrame *pFrameBGR, *swFrame = NULL;    
    int ret, numBytes = 0;
    int  res = -1;

    if (decoder_ctx == NULL) {
        return res;
    }
    fprintf(stderr, "decode_frame 1\n");
    uint8_t *pkbuffer = malloc(pktsize);
    memcpy(pkbuffer, pktdata, pktsize);

    decoder_ctx->width = width;
    decoder_ctx->height = height;
    fprintf(stderr, "decode_frame 2\n");

    decoded_frame = av_frame_alloc();
    if (decoded_frame == NULL) {
        return res;
    }
    av_init_packet(&packet);
    packet.data = (uint8_t *)pkbuffer;
    packet.size = pktsize;

    pFrameBGR = av_frame_alloc();
    swFrame = av_frame_alloc();
    if (NULL == pFrameBGR || NULL == swFrame) {
        goto fail;        
    }
    if (bgrbuffer == NULL){
        numBytes = av_image_get_buffer_size(AV_PIX_FMT_BGR32, decoder_ctx->width, decoder_ctx->height, 1);
        bgrbuffer = (uint8_t *)av_malloc(numBytes * sizeof(uint8_t));        
    }
    av_image_fill_arrays(pFrameBGR->data, pFrameBGR->linesize, bgrbuffer, AV_PIX_FMT_BGR32, decoder_ctx->width, decoder_ctx->height, 1);
    

    ret = avcodec_send_packet(decoder_ctx, &packet);
    if (ret < 0) {        
        goto fail;
    }
    while (ret >= 0) {
        ret = avcodec_receive_frame(decoder_ctx, decoded_frame);        
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
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
            swsContext = sws_getContext(decoder_ctx->width, decoder_ctx->height, AV_PIX_FMT_NV12, decoder_ctx->width, decoder_ctx->height, AV_PIX_FMT_BGR32, SWS_BICUBIC, NULL, NULL, NULL);
        }
        sws_scale(swsContext, (const unsigned char* const*)swFrame->data, swFrame->linesize, 0, decoder_ctx->height, pFrameBGR->data, pFrameBGR->linesize);
#else
        if (swsContext == NULL) {
            swsContext = sws_getContext(decoder_ctx->width, decoder_ctx->height, AV_PIX_FMT_YUVJ420P, decoder_ctx->width, decoder_ctx->height, AV_PIX_FMT_BGR32, SWS_BICUBIC, NULL, NULL, NULL);
            if(swsContext == NULL) {
                fprintf(stderr, "Can not open the scale context\n");
                goto fail;
            }
        }
        sws_scale(swsContext, (const unsigned char* const*)decoded_frame->data, decoded_frame->linesize, 0, decoder_ctx->height, pFrameBGR->data, pFrameBGR->linesize);
#endif        
        pFrameBGR->format = AV_PIX_FMT_BGR32;
        pFrameBGR->width = decoder_ctx->width;
        pFrameBGR->height = decoder_ctx->height;

        memcpy(outdata, bgrbuffer, pFrameBGR->width * pFrameBGR->height * 3);
    }
    res = 0;
fail:
    if(pkbuffer)
        free(pkbuffer);
    av_free(&pFrameBGR);
    av_free(&swFrame);

    if (decoded_frame != NULL)
        av_frame_free(&decoded_frame);
    av_packet_unref(&packet);

    return res;
}