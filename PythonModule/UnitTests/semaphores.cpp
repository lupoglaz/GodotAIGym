#include <boost/interprocess/sync/named_semaphore.hpp>

#include <cstdlib> //std::system
#include <sstream>
#include <iostream>
#include <vector>

#include <unistd.h>
#include <fstream>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <boost/exception/all.hpp>

using namespace boost::interprocess;


class cSharedMemorySemaphore {
    private:
        std::string *name;
        mapped_region *region;
        boost::interprocess::interprocess_semaphore *mutex;
    
    public:
        cSharedMemorySemaphore(){;};
        ~cSharedMemorySemaphore(){
            shared_memory_object::remove(name->c_str());
            delete region;
            delete name;
        };
        void init(const std::string &sem_name){
            
            name = new std::string(sem_name);
            std::cout<<"Constructing semaphore "<<name<<std::endl;
            try{
                shared_memory_object object(open_only, name->c_str(), read_write);
                region = new mapped_region(object, read_write);
            }catch(boost::interprocess::interprocess_exception &e){
                std::cout<<boost::diagnostic_information(e)<<std::endl;
                shared_memory_object::remove(name->c_str());
            }
            std::cout<<"Constructed semaphore "<<name<<std::endl;
        };
        void post(){
            std::cout<<"Post semaphore "<<name<<std::endl;
            mutex = static_cast<interprocess_semaphore*>(region->get_address());
            mutex->post();
        };
        void wait(){
            std::cout<<"Wait semaphore "<<name<<std::endl;
            mutex = static_cast<interprocess_semaphore*>(region->get_address());
            mutex->wait();
        };
};

int main (int argc, char *argv[])
{
    const char *semaphore_name;
    if(argc<2){
        std::cout<<"Segment name not found"<<std::endl;
        return -1;
    }else{
        semaphore_name = argv[1];
    }
    std::cout<<"Segment name = "<<semaphore_name<<std::endl;
    
    try {
        cSharedMemorySemaphore sem;
        sem.init(semaphore_name);

        for(int i=0; i<10; i++){
            sem.wait();
            std::ofstream outfile;
            outfile.open("afile.dat");
            outfile<<i<<std::endl;
            outfile.close();
        }

        return 0;
    } catch (interprocess_exception& e) {
        std::cout << e.what( ) << std::endl;
        return 1;
    }
}