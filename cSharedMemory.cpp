/* cSharedMemory.cpp */

#include "cSharedMemory.h"

#include <core/os/os.h>


#include <cstdlib> //std::system
#include <cstddef>
#include <cassert>
#include <utility>
#include <boost/exception/all.hpp>
#include <exception>
#include <platform/x11_shared/x11_shared.h>

using namespace boost::interprocess;

typedef std::string MyType;

cSharedMemory::cSharedMemory(){
	/*
	try{
		//Open managed segment
		managed_shared_memory segment(open_only, "MySharedMemory");

		//An handle from the base address can identify any byte of the shared
		//memory segment even if it is mapped in different base addresses
		managed_shared_memory::handle_t handle = 0;

		

		//Get buffer local address from handle
		void *msg = segment.get_address_from_handle(handle);

		//Deallocate previously allocated memory
		segment.deallocate(msg);

		std::cout<<std::string( (char*)msg )<<std::endl;

	}catch (boost::interprocess::interprocess_exception &e){
		std::cout<<boost::diagnostic_information(e)<<std::endl;
	}
	*/
	// print_line("Initializing");
	std::cout<<"Initializing"<<std::endl;

	//Obtain handle value
	X11_shared *os = reinterpret_cast<X11_shared*>(OS::get_singleton());
	std::string handle(os->get_shared_handle().str());
	std::cout<<"Handle = "<<handle<<std::endl;
	
	try{//"GodotSharedMemory"
		segment = new managed_shared_memory(open_only, handle.c_str());
	}catch (boost::interprocess::interprocess_exception &e){
		std::cout<<boost::diagnostic_information(e)<<std::endl;
	}
	std::cout<<"Initialized"<<std::endl;
};

cSharedMemory::~cSharedMemory(){
	// print_line("Deinitializing");
	std::cout<<"Deinitializing"<<std::endl;
	delete segment;
	std::cout<<"Done"<<std::endl;
};

int cSharedMemory::get_int(String name){
	std::wstring ws = name.c_str();
	std::string s_name( ws.begin(), ws.end() );
	const char* c_name = s_name.c_str();
	std::cout<<"Getting variable "<<s_name<<std::endl;
	std::pair<int*, managed_shared_memory::size_type> res;
	try{
		std::cout<<"Searching variable "<<s_name<<std::endl;
		res = segment->find<int>(c_name);
		std::cout<<"Done searching, num="<<res.second<<std::endl;
		if(res.second != 1){
			std::cout<<"Instance not found"<<std::endl;
			return NULL;
		}else{
			return *(res.first);
		}
	}catch (boost::interprocess::interprocess_exception &e){
		std::cout<<boost::diagnostic_information(e)<<std::endl;
		return NULL;
	}
	// segment->destroy<MyType>("MyType array");
	
	// return String("Nothing");
	
}
void cSharedMemory::send_variable(){
	//Create an array of 10 elements of MyType initialized to {0.0, 0}
	MyType *array = segment->construct<MyType>
		("MyType array")     //name of the object
		[1]                 //number of elements
		(" ");            //Same two ctor arguments for all objects
}

void cSharedMemory::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_int", "str"), &cSharedMemory::get_int);
	ClassDB::bind_method(D_METHOD("send_variable"), &cSharedMemory::send_variable);
}
