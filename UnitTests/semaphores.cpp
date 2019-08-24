#include <boost/interprocess/sync/named_semaphore.hpp>

#include <cstdlib> //std::system
#include <sstream>
#include <iostream>
#include <vector>

#include <unistd.h>
#include <fstream>

using namespace boost::interprocess;

int main (int argc, char *argv[])
{
    const char *semaphore_name;
    if(argc<2){
        std::cout<<"Segment name not found"<<std::endl;
        return -1;
    }else{
        semaphore_name = argv[1];
    }
    // std::cout<<"Segment name = "<<segment_name<<" handle name to read "<<handle_name_read<<" handle name to wite "<<handle_name_write<<std::endl;
    
    try {
        named_semaphore sem(open_or_create, semaphore_name, 0);

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