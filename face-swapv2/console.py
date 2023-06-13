import argparse
import time

from apps.DeepFaceLive.DeepFaceCSApp import DeepFaceCSApp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="test.flv", help="source face file path")
    parser.add_argument("--target", default="out_test.flv", help="output video file path")
    parser.add_argument("--modelpath", default="models/Jackie_Chan.dfm", help="one to one swap model path")
    parser.add_argument("--swaptype", default = 0, type = int, help="swaptype - 0:face, 1:punkavatar")
    gpuid = 0
    
    args = parser.parse_args()
    
    start_time = time.time()
    app = DeepFaceCSApp(gpuid, args.modelpath)
    elapsed_time = time.time() - start_time
    print("Elapsed time init: ", elapsed_time)

    start_time = time.time()

    app.convert(args.source, args.target, args.swaptype)

    elapsed_time = time.time() - start_time
    print("Elapsed time convert: ", elapsed_time) 

if __name__ == '__main__':
    main()