/* cSharedMemory.cpp */

#include "cSharedMemory.h"

#include <core/os/os.h>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <cstdlib> //std::system
#include <sstream>
#include <iostream>
#include <boost/exception/all.hpp>
#include <exception>
#include <platform/x11_shared/x11_shared.h>

using namespace boost::interprocess;

cSharedMemory::cSharedMemory(){
	try{
		//Open managed segment
		managed_shared_memory segment(open_only, "MySharedMemory");

		//An handle from the base address can identify any byte of the shared
		//memory segment even if it is mapped in different base addresses
		managed_shared_memory::handle_t handle = 0;

		//Obtain handle value
		X11_shared *os = reinterpret_cast<X11_shared*>(OS::get_singleton());
		(os->get_shared_handle()) >> handle;

		//Get buffer local address from handle
		void *msg = segment.get_address_from_handle(handle);

		//Deallocate previously allocated memory
		segment.deallocate(msg);

		std::cout<<std::string( (char*)msg )<<std::endl;

	}catch (boost::interprocess::interprocess_exception &e){
		std::cout<<boost::diagnostic_information(e)<<std::endl;
	}
};

cSharedMemory::~cSharedMemory(){
	
};


void cSharedMemory::_bind_methods() {
	//ClassDB::bind_method(D_METHOD("init", "str", "str", "str"), &PitchDetector::init);
	
}
