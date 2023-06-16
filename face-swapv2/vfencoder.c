#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <libavutil/avassert.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/mathematics.h>
#include <libavutil/timestamp.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>


#define FRAMERATE 8
#define OUT_W 640
#define OUT_H 360

typedef struct OutputStream {
        AVStream* Stream;
        AVCodecContext* Ctx;
        int64_t NextPts;
        AVFrame* Frame;
        AVFrame* bgrFrame;
} OutputStream;

OutputStream VideoSt = { 0 };

AVFormatContext* FmtCtx = NULL;
AVCodec* VCodec = NULL;
AVDictionary* Opt = NULL;
struct SwsContext* SwsCtx = NULL;
AVOutputFormat  *fmt = NULL;
int fourcc = 875967048;
AVPacket Pkt;

void AddStream(enum AVCodecID CodecID)
{
    VCodec = avcodec_find_encoder(CodecID);
    if (!VCodec) {
        fprintf(stderr, "Could not find encoder for '%s'\n",
                avcodec_get_name(CodecID));
        exit(1);
    }
    VideoSt.Stream = avformat_new_stream(FmtCtx, NULL);
    if (!VideoSt.Stream) {
        fprintf(stderr, "Could not allocate stream\n");
        exit(1);
    }
    VideoSt.Stream->id = FmtCtx->nb_streams - 1;
    VideoSt.Ctx = avcodec_alloc_context3(VCodec);
    if (!VideoSt.Ctx) {
        fprintf(stderr, "Could not alloc an encoding context\n");
        exit(1);
    }
    
    VideoSt.Ctx->width = OUT_W;
    VideoSt.Ctx->height = OUT_H;

    VideoSt.Ctx->codec_id = CodecID;
    VideoSt.Ctx->codec_type = AVMEDIA_TYPE_VIDEO;
    VideoSt.Ctx->codec_tag = fourcc;

    VideoSt.Stream->time_base = VideoSt.Ctx->time_base = (AVRational){ 1, FRAMERATE };
    //VideoSt.Ctx->qmin = -1;
    VideoSt.Ctx->gop_size = -1;
    VideoSt.Ctx->max_b_frames = 0;
    VideoSt.Ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    if (VideoSt.Ctx->priv_data) {
        fprintf(stderr, "Alloc priv_data\n");
        av_opt_set(VideoSt.Ctx->priv_data,"crf","23", 0);
    }

    //av_opt_set(VideoSt.Ctx->priv_data, "cq", TCHAR_TO_ANSI(*H264Crf), 0);  // change `cq` to `crf` if using libx264
    //av_opt_set(VideoSt.Ctx->priv_data, "gpu", "0", 0); // comment this line if using libx264

    if (FmtCtx->oformat->flags & AVFMT_GLOBALHEADER)
        VideoSt.Ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
}
void OpenVideoCodec()
{
    int ret;
    AVCodecContext* c = VideoSt.Ctx;
    AVDictionary* tmpopt = NULL;

    av_dict_copy(&tmpopt, Opt, 0);

    ret = avcodec_open2(c, VCodec, NULL);
    av_dict_free(&tmpopt);
    if (ret < 0) {        
        fprintf(stderr, "Could not open video codec: %s\n", av_err2str(ret));
        exit(1);
    }

    VideoSt.Frame = av_frame_alloc();
    if (!VideoSt.Frame) {
        fprintf(stderr, "Could not allocate video frame\n");
        return;
    }

    VideoSt.Frame->format = VideoSt.Ctx->pix_fmt;
    VideoSt.Frame->width = OUT_W;
    VideoSt.Frame->height = OUT_H;

    if (av_frame_get_buffer(VideoSt.Frame, 0) < 0) {
        fprintf(stderr, "Could not allocate frame data\n");
        return;
    }

    VideoSt.bgrFrame = av_frame_alloc();
    if (!VideoSt.bgrFrame) {
        fprintf(stderr, "Could not allocate video frame\n");
        return;
    }

    VideoSt.bgrFrame->format = AV_PIX_FMT_BGR24;
    VideoSt.bgrFrame->width = OUT_W;
    VideoSt.bgrFrame->height = OUT_H;

    if (av_frame_get_buffer(VideoSt.bgrFrame, 24) < 0) {
        fprintf(stderr, "Could not allocate BGR frame data\n");
        return;
    }

    if (avcodec_parameters_from_context(VideoSt.Stream->codecpar, c)) {
        fprintf(stderr, "Could not copy the stream parameters\n");
        exit(1);
    }
}

void CloseStream()
{
    int ret;

    ret = av_write_trailer(FmtCtx);
    if (ret < 0) {
        fprintf(stderr, "writing trailer error: %s\n",av_err2str(ret));
    }

    avcodec_free_context(&VideoSt.Ctx);
    av_frame_free(&VideoSt.Frame);
    av_frame_free(&VideoSt.bgrFrame);
    sws_freeContext(SwsCtx);

    if (!(FmtCtx->oformat->flags & AVFMT_NOFILE))
    {
        ret = avio_closep(&FmtCtx->pb);
        if (ret < 0) {            
            fprintf(stderr, "avio close failed: %s\n",av_err2str(ret));
        }
    }
    avformat_free_context(FmtCtx);
}
void OpenStream(char* filename)
{
    int ret;
    AVCodec *codec;
    int codec_id = AV_CODEC_ID_NONE;    

#if 1
    fmt = av_guess_format(NULL, filename, NULL);
    if (!fmt) {
        printf("Could not av_guess_format.\n");
        return 1;
    }
    codec_id = av_codec_get_id(fmt->codec_tag, fourcc);
    // validate tag
    fourcc = av_codec_get_tag(fmt->codec_tag, codec_id);    
    FmtCtx = avformat_alloc_context();
    //avformat_alloc_output_context2(&FmtCtx, NULL, NULL, filename);
    if (!FmtCtx) {
        printf("Could not deduce output format from file extension.\n");
        return 1;
    }
    /* set file name */
    FmtCtx->oformat = fmt;
    size_t name_len = strlen(filename);
    FmtCtx->url = (char*)av_malloc(name_len + 1);    
    memcpy((void*)FmtCtx->url, filename, name_len + 1);
    FmtCtx->url[name_len] = '\0';
    FmtCtx->max_delay = (int)(0.7*AV_TIME_BASE);
#else
    avformat_alloc_output_context2(&FmtCtx, NULL, NULL, filename);
    if (!FmtCtx) {
        printf("Could not deduce output format from file extension.\n");
        return 1;
    }
#endif    

    /*
    //codec = avcodec_find_encoder_by_name("h264_nvenc");
    codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        fprintf(stderr, "Codec not found\n");
        return 1;
    }
    av_format_set_video_codec(FmtCtx, codec);
    */

    if (FmtCtx->oformat->video_codec != AV_CODEC_ID_NONE) {
        AddStream(FmtCtx->oformat->video_codec);
    }
    OpenVideoCodec();
    VideoSt.NextPts = 0;
    av_dump_format(FmtCtx, 0, filename, 1);

    if (!(FmtCtx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&FmtCtx->pb, filename, AVIO_FLAG_WRITE);
        if (ret < 0) {
            fprintf(stderr, "Could not open '%s': %s\n", filename, av_err2str(ret));
            return 1;
        }
    }
    ret = avformat_write_header(FmtCtx, &Opt);
    if (ret < 0) {
        fprintf(stderr, "Error occurred when opening output file: %s\n", av_err2str(ret));
        return 1;
    }    
    SwsCtx = sws_getContext(VideoSt.Ctx->width, VideoSt.Ctx->height, AV_PIX_FMT_BGR24,
                            VideoSt.Ctx->width, VideoSt.Ctx->height, VideoSt.Ctx->pix_fmt,
                            SWS_BICUBIC, NULL, NULL, NULL);                            
}
void WriteFrame(char* bgrbuffer)
{
    int ret;
    //uint8_t* inData[1];
    //inData[0] = (uint8_t*)bgrbuffer;

    av_init_packet(&Pkt);

    memcpy(VideoSt.bgrFrame->data[0], bgrbuffer, OUT_W*OUT_H*3);

    if(SwsCtx) {
        fprintf(stderr, "Run sws_scale\n");
        sws_scale(SwsCtx, (const unsigned char* const*)VideoSt.bgrFrame->data, VideoSt.bgrFrame->linesize, 0, 
                 VideoSt.Ctx->height, VideoSt.Frame->data, VideoSt.Frame->linesize);
    }
    else {
        fprintf(stderr, "Not created scale context\n");
    }

    fprintf(stderr, "End scale context\n");
    VideoSt.Frame->pts = VideoSt.NextPts++;
    // send the frame to the encoder
    ret = avcodec_send_frame(VideoSt.Ctx, VideoSt.Frame);
    if (ret < 0) {
        fprintf(stderr, "Error sending a frame to the encoder: %s\n",
                av_err2str(ret));
        exit(1);
    }
    while (ret >= 0) {
        ret = avcodec_receive_packet(VideoSt.Ctx, &Pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            break;
        else if (ret < 0) {
            fprintf(stderr, "Error encoding a frame: %s\n", av_err2str(ret));
            exit(1);
        }
        av_packet_rescale_ts(&Pkt, VideoSt.Ctx->time_base, VideoSt.Stream->time_base);
        Pkt.stream_index = VideoSt.Stream->index;
        ret = av_write_frame(FmtCtx, &Pkt);
        if (ret < 0) {
            fprintf(stderr, "Error while writing output packet: %s\n", av_err2str(ret));
            exit(1);
        }
    }
    av_packet_unref(&Pkt);
}

int TextOnePipe(char *filename) 
{
    uint8_t* bgrbuffer = NULL;  
    OpenStream(filename);

    bgrbuffer = (uint8_t*)malloc(OUT_W * OUT_H * 3);
    memset(bgrbuffer, 0xff, OUT_W * OUT_H * 3);    
 
    for(int y=0;y<OUT_H;y++) {
        for(int x=0;x<OUT_W;x++) {
            int base = y*(OUT_W*3);
            bgrbuffer[base + x*3] = 0x00; //B
            bgrbuffer[base + x*3+1] = 0xff;  //G
            bgrbuffer[base + x*3+2] = 0x00;  //R
        }
    }

    for (int i = 0; i < 500; i++)
    {
        WriteFrame(bgrbuffer);
    }
    
    free(bgrbuffer);

    CloseStream();
}
int main(int argc, char **argv)
{
    const char *filename;
      

    if (argc < 2) {
        printf("usage: %s output_file\n", argv[0]);
        return 1;
    }
    filename = argv[1];
    /* allocate the output media context */

    //while (1)
    {
        TextOnePipe(filename);
        //Sleep(1);
    }   

    return 0;
}