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
		std::string arg_s_name(arg_s.begin(), arg_s.end());
		if(arg_s_name.compare(std::string("--handle"))==0){
			arg_s = E->next()->get().c_str();
			std::string val_s( arg_s.begin(), arg_s.end() );
			segment_name = new std::string(val_s);
			found = true;
			print_line(String("Shared memory handle found:") + E->get().c_str() + String(":") + E->next()->get().c_str());
		}
	}
	
	if(!found)return;
	
	try{
		segment = new managed_shared_memory(open_only, segment_name->c_str());
		if(segment == NULL){
			print_line(String("cSharedMemory")+String("Memory segment not found"));	
		}
	}catch (boost::interprocess::interprocess_exception &e){
		print_line(String("cSharedMemory")+String(boost::diagnostic_information(e).c_str()));
		//shared_memory_object::remove(segment_name->c_str());
	}catch(const char *s){
		print_line(String("cSharedMemory")+String(s));
	}
	
	
};

cSharedMemory::~cSharedMemory(){
	//shared_memory_object::remove(segment_name->c_str());
    delete segment;
    delete segment_name;
};

bool cSharedMemory::exists(){
	return found;
}

Ref<cPersistentFloatTensor> cSharedMemory::findFloatTensor(const String &name){
	std::wstring ws = name.c_str();
	std::string s_name( ws.begin(), ws.end() );
	FloatVector *shared_vector = segment->find<FloatVector>(s_name.c_str()).first;
	if(shared_vector == NULL){
		// print_line(String("Not found:")+String(String::num_int64(s_name.length())));
		print_line(String("Not found:")+String(s_name.c_str()));
	}else{
		print_line(String("Found:")+String(s_name.c_str()));
	}
	Ref<cPersistentFloatTensor> tensor(memnew(cPersistentFloatTensor(shared_vector)));
	return tensor;
}
Ref<cPersistentIntTensor> cSharedMemory::findIntTensor(const String &name){
	std::wstring ws = name.c_str();
	std::string s_name( ws.begin(), ws.end() );
	IntVector *shared_vector = segment->find<IntVector> (s_name.c_str()).first;
	if(shared_vector == NULL){
		// print_line(String("Not found:")+String(String::num_int64(s_name.length())));
		print_line(String("Not found:")+String(s_name.c_str()));
	}else{
		print_line(String("Found:")+String(s_name.c_str()));
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
	ClassDB::bind_method(D_METHOD("findIntTensor", "str"), &cSharedMemory::findIntTensor);
	ClassDB::bind_method(D_METHOD("findFloatTensor", "str"), &cSharedMemory::findFloatTensor);
	ClassDB::bind_method(D_METHOD("exists"), &cSharedMemory::exists);
}

void cSharedMemorySemaphore::_bind_methods() {
	ClassDB::bind_method(D_METHOD("post"), &cSharedMemorySemaphore::post);
	ClassDB::bind_method(D_METHOD("wait"), &cSharedMemorySemaphore::wait);
	ClassDB::bind_method(D_METHOD("init", "str"), &cSharedMemorySemaphore::init);
}
