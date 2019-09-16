CC = gcc
CFLAGS = -Wall -Werror -Wmissing-prototypes
OBJS1 = godot_test.o
PROG1 = godot_test
OBJS2 = boost_example.o
PROG2 = boost_example
LOCALLIBDIR = /usr/local/lib
LDFLAGS = -L$(LOCALLIBDIR)
LDLIBS  = -lboost_locale -lboost_system -lstdc++ -lpthread -lrt 

all: $(PROG1) $(PROG2)

$(PROG1): $(OBJS1)
	$(CC) $(CFLAGS) $(OBJS1) -o $@ $(LDFLAGS) $(LDLIBS)

$(PROG2): $(OBJS2)
	$(CC) $(CFLAGS) $(OBJS2) -o $@ $(LDFLAGS) $(LDLIBS)


boost_example.o: boost_example.cpp

godot_test.o: godot_test.cpp

clean:
	rm -f *~ *.o $(PROG) core a.out