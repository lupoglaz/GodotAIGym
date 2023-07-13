/* cSharedMemory.cpp */

#include "shared_memory.h"
#include "godot_cpp/core/error_macros.hpp"

#include <cstdlib>
#include <cstddef>
#include <cassert>
#include <utility>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

using namespace boost::interprocess;
using namespace godot;

cSharedMemory::cSharedMemory(){
	found = false;
};

cSharedMemory::~cSharedMemory(){
	//shared_memory_object::remove(segment_name->c_str());
    delete segment;
};

bool cSharedMemory::init(const String& segment_name){
    try{
		WARN_PRINT( (String("Searching memory segment ") + segment_name).ptr() );
		segment = new managed_shared_memory(open_only, (const char*)segment_name.ptr());
		if(segment == NULL){
            found = false;
			ERR_PRINT( (String("cSharedMemory memory segment not found ") + segment_name).ptr() );
		}
	}catch (boost::interprocess::interprocess_exception &e){
		ERR_PRINT( (String("cSharedMemory")+String(boost::diagnostic_information(e).c_str())).ptr() );
        found = false;
	}catch(const char *s){
		//WARN_PRINT(String("cSharedMemory")+String(s));
        found = false;
	}
    return found;
}

bool cSharedMemory::exists(){
	return found;
}

Ref<cPersistentFloatTensor> cSharedMemory::findFloatTensor(const String &name){
	FloatVector *shared_vector = segment->find<FloatVector>((const char*)name.ptr()).first;
	if(shared_vector == NULL){
		ERR_PRINT( (String("Not found:")+name).ptr() );
	}else{
		//WARN_PRINT(String("Found:")+String(s_name.c_str()));
	}
	Ref<cPersistentFloatTensor> tensor(memnew(cPersistentFloatTensor(shared_vector)));
	return tensor;
}
Ref<cPersistentIntTensor> cSharedMemory::findIntTensor(const String &name){
	IntVector *shared_vector = segment->find<IntVector> ((const char*)name.ptr()).first;
	if(shared_vector == NULL){
		ERR_PRINT( (String("Not found:")+String(name)).ptr() );
	}else{
		//print_line(String("Found:")+String(s_name.c_str()));
	}
	Ref<cPersistentIntTensor> tensor(memnew(cPersistentIntTensor(shared_vector)));
	return tensor;
}

void cPersistentFloatTensor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("read"), &cPersistentFloatTensor::read);
	ClassDB::bind_method(D_METHOD("write", "array"), &cPersistentFloatTensor::write);
}
void cPersistentIntTensor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("read"), &cPersistentIntTensor::read);
	ClassDB::bind_method(D_METHOD("write", "array"), &cPersistentIntTensor::write);
}

void cSharedMemory::_bind_methods() {
    ClassDB::bind_method(D_METHOD("init", "str"), &cSharedMemory::init);
	ClassDB::bind_method(D_METHOD("findIntTensor", "str"), &cSharedMemory::findIntTensor);
	ClassDB::bind_method(D_METHOD("findFloatTensor", "str"), &cSharedMemory::findFloatTensor);
	ClassDB::bind_method(D_METHOD("exists"), &cSharedMemory::exists);
}

void cSharedMemorySemaphore::_bind_methods() {
	ClassDB::bind_method(D_METHOD("post"), &cSharedMemorySemaphore::post);
	ClassDB::bind_method(D_METHOD("wait"), &cSharedMemorySemaphore::wait);
	ClassDB::bind_method(D_METHOD("init", "str"), &cSharedMemorySemaphore::init);
}
