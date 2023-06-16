package main

import (
	"flag"
	"fmt"
	"net"
	"net/url"
	"os"
	"time"

	"github.com/yapingcat/gomedia/go-codec"
	"github.com/yapingcat/gomedia/go-flv"
	"github.com/yapingcat/gomedia/go-rtmp"
)

var rtmpUrl = flag.String("url", "rtmp://127.0.0.1/live/faceswap", "publish rtmp url")
var flvFile = flag.String("flv", "1.flv", "push flv file to server")

func publish(fileName string, cli *rtmp.RtmpClient) {
	f := flv.CreateFlvReader()
	f.OnFrame = func(cid codec.CodecID, frame []byte, pts, dts uint32) {
		if cid == codec.CODECID_VIDEO_H264 {
			cli.WriteVideo(cid, frame, pts, dts)
			time.Sleep(time.Millisecond * 33)
		} else if cid == codec.CODECID_AUDIO_AAC {
			cli.WriteAudio(cid, frame, pts, dts)
		}
	}
	fd, _ := os.Open(fileName)
	defer fd.Close()
	cache := make([]byte, 4096)
	for {
		n, err := fd.Read(cache)
		if err != nil {
			fmt.Println(err)
			break
		}
		f.Input(cache[0:n])
	}
}

func main() {
	flag.Parse()
	u, err := url.Parse(*rtmpUrl)
	if err != nil {
		panic(err)
	}
	host := u.Host
	if u.Port() == "" {
		host += ":1935"
	}
	//connect to remote rtmp server
	connect, err := net.Dial("tcp4", host)
	if err != nil {
		fmt.Println("connect failed", err)
		return
	}

	isReady := make(chan struct{})

	//create rtmp client
	cli := rtmp.NewRtmpClient(rtmp.WithComplexHandshake(), rtmp.WithEnablePublish())

	//monotoring status ,STATE_RTMP_PUBLISH_START mean ready to receive
	cli.OnStateChange(func(newState rtmp.RtmpState) {
		if newState == rtmp.STATE_RTMP_PUBLISH_START {
			fmt.Println("ready for publish")
			close(isReady)
		}
	})

	cli.SetOutput(func(data []byte) error {
		_, err := connect.Write(data)
		return err
	})

	go func() {
		<-isReady
		fmt.Println("start to read flv")
		for i := 0; i < 50; i++ {
			publish(*flvFile, cli)
		}
		fmt.Println("end publishing")
	}()

	fmt.Println("cli start")
	cli.Start(*rtmpUrl)
	buf := make([]byte, 4096)
	n := 0
	for err == nil {
		n, err = connect.Read(buf)
		fmt.Println("Read cli")
		if err != nil {
			continue
		}
		cli.Input(buf[:n])
	}
	fmt.Println(err)
	fmt.Println("end rtmptest program")
}
