/* cSharedMemory.h */
#ifndef SUMMATOR_H
#define SUMMATOR_H

#include "core/reference.h"
#include "core/pool_vector.h"

#include <iostream>
#include <string>
#include <vector>
#include <exception>

#include <torch/script.h>
using namespace torch::indexing;

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <boost/exception/all.hpp>

using namespace boost::interprocess;

typedef allocator<int, managed_shared_memory::segment_manager>  ShmemAllocator;
typedef std::vector<int, ShmemAllocator> IntVector;
typedef std::vector<float, ShmemAllocator> FloatVector;


struct TensorDescription{
    std::string type;       //Tensor scalar type
    TensorDescription(const std::string &t_type):type(t_type){}
};

class cSharedMemory : public Reference {
    GDCLASS(cSharedMemory, Reference);

private:
    std::string *segment_name = NULL;
    managed_shared_memory *segment = NULL;
    shared_memory_object *object = NULL;

    bool found;

protected:
    static void _bind_methods();

public:
    cSharedMemory();
    ~cSharedMemory();

    PoolVector<int> getIntArray(const String &name);
    PoolVector<float> getFloatArray(const String &name);
    void sendIntArray(const String &name, const PoolVector<int> &array);
    void sendFloatArray(const String &name, const PoolVector<float> &array);
    bool exists();
};


class cSharedMemorySemaphore : public Reference {
    GDCLASS(cSharedMemorySemaphore, Reference);
    private:
        std::string *name;
        mapped_region *region;
        interprocess_semaphore *mutex;
    
    protected:
        static void _bind_methods();
    
    public:
        cSharedMemorySemaphore(){;};
        ~cSharedMemorySemaphore(){
            shared_memory_object::remove(name->c_str());
            delete region;
            delete name;
        };
        void init(const String &sem_name){
            
            std::wstring ws = sem_name.c_str();
	        std::string s_name( ws.begin(), ws.end() );
            name = new std::string(s_name);
            // std::cout<<"Constructing semaphore "<<*name<<std::endl;
            try{
                shared_memory_object object(open_only, name->c_str(), read_write);
                region = new mapped_region(object, read_write);
            }catch(interprocess_exception &e){
                std::cout<<*name<<":"<<boost::diagnostic_information(e)<<std::endl;
                shared_memory_object::remove(name->c_str());
            }
            // std::cout<<"Constructed semaphore "<<*name<<std::endl;
        };
        void post(){
            // std::cout<<"Post semaphore "<<*name<<std::endl;
            mutex = static_cast<interprocess_semaphore*>(region->get_address());
            mutex->post();
            // std::cout<<"Posted semaphore "<<*name<<std::endl;
        };
        void wait(){
            // std::cout<<"Wait semaphore "<<*name<<std::endl;
            mutex = static_cast<interprocess_semaphore*>(region->get_address());
            mutex->wait();
            // std::cout<<"Waited semaphore "<<*name<<std::endl;
        };
};

class cTorchModel : public Reference{
    GDCLASS(cTorchModel, Reference);

    private:
        torch::jit::script::Module module;

    protected:
        static void _bind_methods();

    public:
        cTorchModel(){;};
        ~cTorchModel(){;};

        void load(String path){
            this->module = torch::jit::load(path.ascii().get_data());
        };
        PoolVector<float> run(const PoolVector<float> &input){
            PoolVector<float> output;
            at::Tensor input_t = torch::zeros({1, input.size()}, torch::TensorOptions().dtype(torch::kFloat32));
            for(int i=0; i<input.size(); i++) input_t.index_put_({0,i}, input[i]);
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_t);
            at::Tensor output_t = this->module.forward(inputs).toTensor();
            auto output_t_a = output_t.accessor<float,2>();
            for(int i=0; i<output_t.sizes()[1]; i++) output.push_back(output_t_a[0][i]);
            return output;
        }

};

#endif