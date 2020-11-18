/* cSharedMemory.cpp */

#include "cSharedMemory.h"

#include <core/os/os.h>

#include <cstdlib> //std::system
#include <cstddef>
#include <cassert>
#include <utility>
#include <platform/x11/os_x11.h>
// #include <platform/x11_shared/x11_shared.h>

using namespace boost::interprocess;

typedef std::string MyType;

cSharedMemory::cSharedMemory(){
	//Obtain segment_name value
	OS_X11 *os = reinterpret_cast<OS_X11*>(OS::get_singleton());
	
	found = false;
	std::wstring arg_s;
	for(List<String>::Element *E=os->get_cmdline_args().front(); E; E=E->next()) {
		arg_s = E->get().c_str();
		std::string arg_s_name( arg_s.begin(), arg_s.end() );
		if(arg_s_name.compare(std::string("--handle"))==0){
			arg_s = E->next()->get().c_str();
			std::string val_s( arg_s.begin(), arg_s.end() );
			segment_name = new std::string(val_s);
			found = true;
			print_line(String("Shared memory handle found:") + E->get() + String(":") + E->next()->get());
		}
	}
	
	if(!found)return;
	
	try{
		segment = new managed_shared_memory(open_only, segment_name->c_str());
	}catch (boost::interprocess::interprocess_exception &e){
		std::cout<<boost::diagnostic_information(e)<<std::endl;
		shared_memory_object::remove(segment_name->c_str());
	}
	
};

cSharedMemory::~cSharedMemory(){
	shared_memory_object::remove(segment_name->c_str());
    delete segment;
    delete segment_name;
};

bool cSharedMemory::exists(){
	return found;
}

PoolVector<int> cSharedMemory::getIntArray(const String &name){
	std::wstring ws = name.c_str();
	std::string s_name( ws.begin(), ws.end() );
	
	PoolVector<int> data;
	try{
		IntVector *shared_vector = segment->find<IntVector> (s_name.c_str()).first;
		for(int i=0; i<shared_vector->size(); i++){
			data.push_back( (*shared_vector)[i] );
		}
		segment->destroy<IntVector>(s_name.c_str());
	}catch(interprocess_exception &ex){
        std::cout<<s_name<<":"<<ex.what()<<std::endl;
    }catch(std::exception &ex){
        std::cout<<ex.what()<<std::endl;
    }catch(const char *s){
		std::cout<<s<<std::endl;
	}
	return data;
}

PoolVector<float> cSharedMemory::getFloatArray(const String &name){
	std::wstring ws = name.c_str();
	std::string s_name( ws.begin(), ws.end() );
	
	PoolVector<float> data;
	try{
		FloatVector *shared_vector = segment->find<FloatVector> (s_name.c_str()).first;
		for(int i=0; i<shared_vector->size(); i++){
			data.push_back( (*shared_vector)[i] );
		}
		segment->destroy<FloatVector>(s_name.c_str());
	}catch(interprocess_exception &ex){
        std::cout<<s_name<<":"<<ex.what()<<std::endl;
    }catch(std::exception &ex){
        std::cout<<ex.what()<<std::endl;
    }catch(const char *s){
		std::cout<<s<<std::endl;
	}
	return data;
}

void cSharedMemory::sendIntArray(const String &name, const PoolVector<int> &array){
	std::wstring ws = name.c_str();
	std::string s_name( ws.begin(), ws.end() );
	try{
		const ShmemAllocator alloc_inst (segment->get_segment_manager());
		IntVector *shared_vector = segment->construct<IntVector>(s_name.c_str())(alloc_inst);
		for(int i=0; i<array.size(); i++)
			shared_vector->push_back(array[i]);

	}catch(interprocess_exception &ex){
        std::cout<<s_name<<":"<<ex.what()<<std::endl;
    }catch(std::exception &ex){
        std::cout<<ex.what()<<std::endl;
    }
}

void cSharedMemory::sendFloatArray(const String &name, const PoolVector<float> &array){
	std::wstring ws = name.c_str();
	std::string s_name( ws.begin(), ws.end() );
	try{
		const ShmemAllocator alloc_inst (segment->get_segment_manager());
		FloatVector *shared_vector = segment->construct<FloatVector>(s_name.c_str())(alloc_inst);
		for(int i=0; i<array.size(); i++)
			shared_vector->push_back(array[i]);
	}catch(interprocess_exception &ex){
        std::cout<<s_name<<":"<<ex.what()<<std::endl;
    }catch(std::exception &ex){
        std::cout<<ex.what()<<std::endl;
    }
}

void cSharedMemory::_bind_methods() {
	ClassDB::bind_method(D_METHOD("getIntArray", "str"), &cSharedMemory::getIntArray);
	ClassDB::bind_method(D_METHOD("getFloatArray", "str"), &cSharedMemory::getFloatArray);
	ClassDB::bind_method(D_METHOD("sendIntArray", "str", "array"), &cSharedMemory::sendIntArray);
	ClassDB::bind_method(D_METHOD("sendFloatArray", "str", "array"), &cSharedMemory::sendFloatArray);
	ClassDB::bind_method(D_METHOD("exists"), &cSharedMemory::exists);
}

void cSharedMemorySemaphore::_bind_methods() {
	ClassDB::bind_method(D_METHOD("post"), &cSharedMemorySemaphore::post);
	ClassDB::bind_method(D_METHOD("wait"), &cSharedMemorySemaphore::wait);
	ClassDB::bind_method(D_METHOD("init", "str"), &cSharedMemorySemaphore::init);
}