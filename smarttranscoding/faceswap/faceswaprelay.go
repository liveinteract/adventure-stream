package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/woody0105/interactive-video/smarttranscoding/ffmpeg"
	"github.com/yapingcat/gomedia/go-codec"
	"github.com/yapingcat/gomedia/go-flv"
	"github.com/yapingcat/gomedia/go-rtmp"
)

//var nodes = []string{"127.0.0.1", "35.194.58.82", "34.135.170.77"}
var addr = flag.String("addr", "localhost:8002", "http service address")
var nodes = []string{"127.0.0.1"}
var isFirstFrame = true

//variable for rtmp communication
var rtmpUrl = flag.String("rtmpurl", "rtmp://127.0.0.1/live/faceswap", "rtmp url to publish")
var flvname string = ""
var flvfile *os.File = nil
var flvmutex sync.Mutex

var rtmpcli *rtmp.RtmpClient = nil
var rtmpendflag = false
var packetlen int = 0
var packetoffsets = []int{}
var framewidth int = 0
var frameheight int = 0

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

func handleconnections1(w http.ResponseWriter, r *http.Request) {

	codec := r.Header.Get("X-WS-Codec")
	size := r.Header.Get("X-WS-Video-Size")
	sizetmp := strings.Split(size, "x")
	width, _ := strconv.Atoi(sizetmp[0])
	height, _ := strconv.Atoi(sizetmp[1])

	framewidth = width
	frameheight = height

	respheader := make(http.Header)
	initData := r.Header.Get("X-Ws-Init")
	spsData, _ := base64.StdEncoding.DecodeString(initData)

	// var instances []ffmpeg.Instance
	/*
		facebuf1, _ := ioutil.ReadFile("example/faces/face1")
		facebuf2, _ := ioutil.ReadFile("example/faces/face2")
		facebuf3, _ := ioutil.ReadFile("example/faces/philipp")
		facebuf4, _ := ioutil.ReadFile("example/faces/eric")

		metadata := fmt.Sprintf(`
		[
			{"id": "1", "name": "Nick", "image": "%s", "metadata": "https://test.org/dev1", "action": "embedlink"},
			{"id": "2", "name": "James", "image": "%s", "metadata": "https://test.org/dev2", "action": "embedlink"},
			{"id": "3", "name": "Philipp", "image": "%s", "metadata": "https://test.org/Philipp", "action": "embedlink"},
			{"id": "4", "name": "Eric", "image": "%s", "metadata": "https://test.org/Eric", "action": "embedlink"}
		]
		`, string(facebuf1), string(facebuf2), string(facebuf3), string(facebuf4))

		resp, err := ffmpeg.RegisterSamples(bytes.NewBuffer([]byte(metadata)))

		if err != nil {
			log.Print("instance registration failure", err)
		} else {
			respBody, _ := io.ReadAll(resp.Body)
			fmt.Println(string(respBody))
		}
	*/

	respheader.Add("Sec-WebSocket-Protocol", "videoprocessing.livepeer.com")
	c, err := upgrader.Upgrade(w, r, respheader)
	if err != nil {
		log.Print("upgrade:", err)
		return
	}
	defer c.Close()

	fmt.Println("video codec id:", codec, width, height)

	ffmpeg.SetDecoderCtxParams(width, height)
	handlemsg1(w, r, c, codec, spsData)

}

func handlemsg1(w http.ResponseWriter, r *http.Request, conn *websocket.Conn, codec string, initData []byte) {
	for {
		_, message, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("error: %v", err)
			}
			log.Printf("read:%v", err)
			conn.Close()
			isFirstFrame = true
			break
		}
		timestamp := binary.BigEndian.Uint64(message[:8])
		packetdata := message[8:]

		if isFirstFrame {
			fmt.Println("sps packet, appending initData", initData)
			packetdata = append(initData, packetdata...)
			isFirstFrame = false
			if rtmpendflag {
				go startrtmpClient()
			}
		}

		timedpacket := ffmpeg.TimedPacket{Timestamp: timestamp, Packetdata: ffmpeg.APacket{Data: packetdata, Length: len(packetdata)}}
		//ffmpeg.FeedPacket(timedpacket, nodes, conn, nodes)
		ProcessingFswap(timedpacket)
	}
}

func startServer1() {
	log.Println("started server", *addr)
	http.HandleFunc("/segmentation", handleconnections1)
}

// part of RTMP communication
type SwapRes struct {
	Respath string `json:"respath"`
}

func callpushrtmp(fileName string, packetStr string) {
	client := &http.Client{}
	//nodelen := len(nodes)
	url := fmt.Sprintf("http://%s:5555/faceswap", nodes[0])
	metadata := fmt.Sprintf(`{"filepath": "%s", "pktinfo": "%s", "w": "%d", "h": "%d"}`,
		fileName, packetStr, framewidth, frameheight)

	req, _ := http.NewRequest("POST", url, bytes.NewBuffer([]byte(metadata)))
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		log.Println(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		bodyBytes, err := io.ReadAll(resp.Body)
		if err == nil {
			var res SwapRes
			err = json.Unmarshal(bodyBytes, &res)
			if len(res.Respath) > 0 {
				log.Printf("Json Result %s", res.Respath)
				flvmutex.Lock()
				pushrtmp(res.Respath, rtmpcli)
				flvmutex.Unlock()
				err = os.Remove(res.Respath)
				if err != nil {
					fmt.Println("Error: ", err) //print the error if file is not removed
				}
			}
		} else {
			log.Println(err)
		}

	} else {
		log.Printf("url: %s\n status code %d", url, resp.StatusCode)
	}
	os.Remove(fileName)
}
func pushrtmp(fileName string, cli *rtmp.RtmpClient) {
	if cli == nil {
		log.Println("cli - not nil point")
		return
	}

	f := flv.CreateFlvReader()
	f.OnFrame = func(cid codec.CodecID, frame []byte, pts, dts uint32) {
		if cid == codec.CODECID_VIDEO_H264 {
			cli.WriteVideo(cid, frame, pts, dts)
			time.Sleep(time.Millisecond * 1)
		} else if cid == codec.CODECID_AUDIO_AAC {
			cli.WriteAudio(cid, frame, pts, dts)
		}
	}
	fd, _ := os.Open(fileName)
	//defer fd.Close()
	cache := make([]byte, 8182)
	for {
		n, err := fd.Read(cache)
		if err != nil {
			break
		}
		f.Input(cache[0:n])
	}
	fd.Close()
}
func ProcessingFswap(pkt ffmpeg.TimedPacket) {
	// if starting
	data := pkt.Packetdata.Data
	if packetlen == 0 && flvfile == nil {
		unixMilli := time.Now().UnixNano() / 1e6
		flvname = strconv.Itoa(int(unixMilli)) + ".data"

		path, _ := os.Getwd()
		flvname = filepath.Join(path, flvname)
		flvfile, _ = os.OpenFile(flvname, os.O_CREATE|os.O_RDWR, 0666)
	}

	flvfile.Write(data)
	packetlen += len(data)
	packetoffsets = append(packetoffsets, len(data))

	//if packetlen > 50000 {
	if packetlen > 50000 {
		flvfile.Close()
		flvfile = nil
		packetlen = 0
		packetStr := ""
		for _, str := range packetoffsets {
			packetStr += (strconv.Itoa(str) + ",")
		}
		packetoffsets = nil

		go callpushrtmp(flvname, packetStr)
	}
}
func startrtmpClient() {
	rtmpendflag = false
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
	rtmpcli = rtmp.NewRtmpClient(rtmp.WithComplexHandshake(), rtmp.WithEnablePublish())

	//monotoring status ,STATE_RTMP_PUBLISH_START mean ready to receive
	rtmpcli.OnStateChange(func(newState rtmp.RtmpState) {
		if newState == rtmp.STATE_RTMP_PUBLISH_START {
			fmt.Println("ready for publish")
			close(isReady)
		}
	})

	rtmpcli.SetOutput(func(data []byte) error {
		_, err := connect.Write(data)
		return err
	})

	fmt.Println("rtmpcli start")
	rtmpcli.Start(*rtmpUrl)
	buf := make([]byte, 4096)
	n := 0
	for err == nil {
		n, err = connect.Read(buf)
		if err != nil {
			continue
		}
		rtmpcli.Input(buf[:n])
	}
	fmt.Println(err)
	fmt.Println("Ending rtmpcli")
	rtmpendflag = true
}
func main() {
	flag.Parse()
	log.SetFlags(0)
	ffmpeg.DecoderInit()
	go startrtmpClient()
	startServer1()
	log.Fatal(http.ListenAndServe(*addr, nil))
}
