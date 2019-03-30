/* cSharedMemory.h */
#ifndef SUMMATOR_H
#define SUMMATOR_H

#include "core/reference.h"

#include <iostream>
#include <string>
#include <exception>
#include <math.h>
#include <boost/interprocess/managed_shared_memory.hpp>

using namespace boost::interprocess;

class cSharedMemory : public Reference {
    GDCLASS(cSharedMemory, Reference);

private:
    managed_shared_memory *segment;

protected:
    static void _bind_methods();

public:
    cSharedMemory();
    ~cSharedMemory();

    int get_int(String name);
    void send_variable();
};

#endif