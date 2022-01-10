#include "cGodotSharedInterface.h"
#include <sstream>

cSharedMemoryTensor::cSharedMemoryTensor(const std::string &name){
	segment_name = new std::string(name);
	try{
		//Create a managed shared memory segment
		shared_memory_object::remove(segment_name->c_str());
		segment = new managed_shared_memory(create_only, segment_name->c_str(), 65536);
	}catch (interprocess_exception& ex) {
	std::cout<<"PythonModule:cSharedMemoryTensor:"<<name<<":"<<boost::diagnostic_information(ex)<<ex.get_native_error()<<std::endl;
		shared_memory_object::remove(segment_name->c_str());
	}
}
cSharedMemoryTensor::~cSharedMemoryTensor(){
	shared_memory_object::remove(segment_name->c_str());
	delete segment;
	delete segment_name;
}

cPersistentIntTensor* cSharedMemoryTensor::newIntTensor(const std::string &name, int size){
	const ShmemAllocatorInt alloc_inst(segment->get_segment_manager());
	IntVector *myvector = segment->construct<IntVector>(name.c_str())(alloc_inst);
	if(myvector == NULL){
		std::cout<<"Cannot create vector "<<name<<std::endl;
	}else{
		// std::cout<<"Created vector "<<name.length()<<std::endl;
		std::cout<<"Created int32 vector "<<name<<" size = "<<size<<std::endl;
	}
	myvector->resize(size);
	cPersistentIntTensor *tensor = new cPersistentIntTensor(myvector, name, segment);
	return tensor;
}

cPersistentFloatTensor* cSharedMemoryTensor::newFloatTensor(const std::string &name, int size){
	const ShmemAllocatorFloat alloc_inst(segment->get_segment_manager());
	FloatVector *myvector = segment->construct<FloatVector>(name.c_str())(alloc_inst);
	if(myvector == NULL){
		std::cout<<"Cannot create vector "<<name<<std::endl;
	}else{
		// std::cout<<"Created vector "<<name.length()<<std::endl;
		std::cout<<"Created float32 vector "<<name<<" size = "<<size<<std::endl;
	}
	myvector->resize(size);
	cPersistentFloatTensor *tensor = new cPersistentFloatTensor(myvector, name, segment);
	return tensor;
}

cSharedMemorySemaphore::cSharedMemorySemaphore(const std::string &sem_name, int init_count){
	try{
		name = new std::string(sem_name);
		shared_memory_object::remove(name->c_str());
		shared_memory_object object(open_or_create, name->c_str(), read_write);
		object.truncate(sizeof(interprocess_semaphore));
		region = new mapped_region(object, read_write);
		void *addr = region->get_address();
		mutex = new (addr) interprocess_semaphore(init_count);
	}catch(interprocess_exception &ex){
		std::cout<<"PythonModule:"<<sem_name<<":"<<boost::diagnostic_information(ex)<<ex.get_native_error()<<std::endl;
	}catch(std::exception &ex){
		std::cout<<"PythonModule:"<<ex.what()<<std::endl;
	}

}

cSharedMemorySemaphore::~cSharedMemorySemaphore(){
	shared_memory_object::remove(name->c_str());
	delete region;
	delete name;
	delete mutex;
}

void cSharedMemorySemaphore::post(){
	// std::cout<<"Post semaphore "<<*name<<std::endl;
	try{
		//Important: if the mutex is not cast here, it will give segfault
		mutex = static_cast<interprocess_semaphore*>(region->get_address());
		mutex->post();
	}catch(boost::interprocess::interprocess_exception &ex){
		std::cout<<"PythonModule:post:"<<*name<<":"<<boost::diagnostic_information(ex)<<ex.get_native_error()<<std::endl;
	}catch(std::exception &ex){
		std::cout<<"PythonModule:post:"<<ex.what()<<std::endl;
	}
	// std::cout<<"Posted semaphore "<<*name<<std::endl;
}
void cSharedMemorySemaphore::wait(){
	// std::cout<<"wait semaphore "<<*name<<std::endl;
	try{
		mutex = static_cast<interprocess_semaphore*>(region->get_address());
		mutex->wait();
	}catch(boost::interprocess::interprocess_exception &ex){
		std::cout<<"PythonModule:wait:"<<*name<<":"<<boost::diagnostic_information(ex)<<ex.get_native_error()<<std::endl;
	}catch(std::exception &ex){
		std::cout<<"PythonModule:wait:"<<ex.what()<<std::endl;
	}
	// std::cout<<"wait semaphore "<<*name<<std::endl;
}
void cSharedMemorySemaphore::timed_wait(int time){
	// std::cout<<"wait semaphore "<<*name<<std::endl;
	try{
		boost::posix_time::ptime timeout(boost::posix_time::microsec_clock::universal_time()+boost::posix_time::millisec(time));
		mutex = static_cast<interprocess_semaphore*>(region->get_address());
		mutex->timed_wait(timeout);
	}catch(boost::interprocess::interprocess_exception &ex){
		std::cout<<"PythonModule:timed_wait:"<<*name<<":"<<boost::diagnostic_information(ex)<<ex.get_native_error()<<std::endl;
	}catch(std::exception &ex){
		std::cout<<"PythonModule:timed_wait:"<<ex.what()<<std::endl;
	}
	// std::cout<<"wait semaphore "<<*name<<std::endl;
}
