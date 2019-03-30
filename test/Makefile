CC = gcc
CFLAGS = -Wall -Werror -Wmissing-prototypes
OBJS = boost_example.o
PROG = boost_example
LOCALLIBDIR = /usr/local/lib
LDFLAGS = -L$(LOCALLIBDIR)
LDLIBS  = -lboost_locale -lboost_system -lstdc++ -lpthread -lrt 

all: $(PROG)

$(PROG): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS) $(LDLIBS)

boost_example.o: boost_example.cpp

clean:
	rm -f *~ *.o $(PROG) core a.out