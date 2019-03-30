/* cSharedMemory.h */
#ifndef SUMMATOR_H
#define SUMMATOR_H

#include "core/reference.h"

#include <iostream>
#include <string>
#include <exception>
#include <math.h>

class cSharedMemory : public Reference {
    GDCLASS(cSharedMemory, Reference);

private:
        

protected:
    static void _bind_methods();

public:
    cSharedMemory();
    ~cSharedMemory();
};

#endif