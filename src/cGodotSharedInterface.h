#include <torch/extension.h>
#include <string>
#include <boost/interprocess/managed_shared_memory.hpp>

struct Pet {
    Pet(const std::string &name) : name(name) { }
    void setName(const std::string &name_) { name = name_; }
    const std::string &getName() const { return name; }

    std::string name;
};

int add(int i, int j);

using namespace boost::interprocess;

class cSharedMemoryTensor{

    private:
        std::string *segment_name;
        managed_shared_memory *segment;        
        void * shptr;
    public:
        cSharedMemoryTensor(const std::string &name);
        ~cSharedMemoryTensor();

        std::string getHandle() const;  
};