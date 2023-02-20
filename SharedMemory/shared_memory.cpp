/* cSharedMemory.cpp */

#include "shared_memory.h"

#include <cstdlib>
#include <cstddef>
#include <cassert>
#include <utility>

using namespace boost::interprocess;

cSharedMemory::cSharedMemory(){
	found = false;
};

cSharedMemory::~cSharedMemory(){
	//shared_memory_object::remove(segment_name->c_str());
    delete segment;
};

bool cSharedMemory::init(const String& segment_name){
    try{
		segment = new managed_shared_memory(open_only, (const char*)segment_name.ptr());
		if(segment == NULL){
            found = false;
			//ERR_FAIL_MSG(String("cSharedMemory memory segment not found"));
		}
	}catch (boost::interprocess::interprocess_exception &e){
		//ERR_FAIL_MSG(String("cSharedMemory")+String(boost::diagnostic_information(e).c_str()));
        found = false;
	}catch(const char *s){
		//print_line(String("cSharedMemory")+String(s));
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
		//ERR_FAIL_MSG(String("Not found:")+String(s_name.c_str()));
	}else{
		//print_line(String("Found:")+String(s_name.c_str()));
	}
	Ref<cPersistentFloatTensor> tensor(memnew(cPersistentFloatTensor(shared_vector)));
	return tensor;
}
Ref<cPersistentIntTensor> cSharedMemory::findIntTensor(const String &name){
	IntVector *shared_vector = segment->find<IntVector> ((const char*)name.ptr()).first;
	if(shared_vector == NULL){
		//ERR_FAIL_MSG(String("Not found:")+String(s_name.c_str()));
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
