CC = gcc  # C compiler
FLAGS = -I$(HOME)/compiled/include
CFLAGS = -fPIC -Wall -Wextra -O2 -g
LDFLAGS = -shared   # linking flags
DEBUGFLAGS   = -O0 -D _DEBUG
RELEASEFLAGS = -O2 -D NDEBUG -combine -fwhole-program

TARGET  = fvdecoder.so
SOURCES = $(shell echo ./*.c)
OBJECTS = $(SOURCES:.c=.o)

PREFIX = $(HOME)/compiled
BINDIR = $(HOME)/lib

all: $(TARGET)

$(TARGET): $(OBJECTS)
    $(CC) $(FLAGS) $(CFLAGS) $(LDFLAGS) -o $(TARGET) $(OBJECTS)

gcc -c -Wall -fpic -I ~/compiled/include vfdecoder.c
gcc -shared -L ~/compiled/lib -o fvdecoder.so vfdecoder.o -lavformat -lavfilter -lavcodec -lavutil -lswscale
