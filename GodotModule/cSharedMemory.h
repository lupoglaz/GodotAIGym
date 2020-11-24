/* cSharedMemory.h */
#ifndef SUMMATOR_H
#define SUMMATOR_H

#include "core/reference.h"
#include "core/pool_vector.h"
#include "core/io/resource_loader.h"
#include "core/variant_parser.h"
#include "core/project_settings.h"


#include <string>
#include <vector>
#include <exception>
#include <iostream>
#include <istream>
#include <streambuf>

#include <torch/script.h>
#include <torch/csrc/jit/serialization/export.h>
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


class cTorchModelData : public Resource{
    GDCLASS(cTorchModelData, Resource);

    private:
        PoolByteArray content;
        
    protected:
        static void _bind_methods(){
        	ClassDB::bind_method(D_METHOD("set_array", "array"), &cTorchModelData::set_array);
            ClassDB::bind_method(D_METHOD("get_array"), &cTorchModelData::get_array);
            ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "array"), "set_array", "get_array");
        }

    public:
        void load(const char *_data, size_t size){
            print_line("Converting string to PoolByteArray");
            //Convert model to poolbytearray
            content.resize(size);
            for(long i=0; i<size; i++)
                content.set(i,_data[i]);
        };
        long size() const{
            return content.size();
        }
        void save(char *_data) const{
            print_line("Converting PoolByteArray to string");
            for(int i=0; i<content.size(); i++)
                _data[i] = content[i];
        }

        void set_array(const PoolByteArray &p_array){
            content = p_array;
        };
        PoolByteArray get_array() const{
            return content;
        };

        

};

class cTorchModelLoader : public ResourceFormatLoader {
    GDCLASS(cTorchModelLoader, ResourceFormatLoader);
public:
    virtual RES load(const String &p_path, const String &p_original_path, Error *r_error = NULL){
        print_line("Loading from file to ModelData");
        //Load file in the resources to a model
        if (r_error) {
            *r_error = OK;
        }
        String r_path;
        if (ProjectSettings::get_singleton()) {
        	if (p_path.begins_with("res://")) {
        		String resource_path = ProjectSettings::get_singleton()->get_resource_path();
        		if (resource_path != "") {
        			r_path = p_path.replace("res:/", resource_path);
        		}
        		r_path = p_path.replace("res://", "");
        	}
        }
        torch::jit::script::Module module = torch::jit::load(r_path.ascii().get_data());
        std::stringstream str;
        torch::jit::ExportModule(module, str);
        Ref<cTorchModelData> model_data = memnew(cTorchModelData);
        model_data->load(str.str().c_str(), str.str().size());
        return model_data;
    }
    virtual void get_recognized_extensions(List<String> *r_extensions) const{
        if (!r_extensions->find("jit")) {
            r_extensions->push_back("jit");
        }
    }
    virtual bool handles_type(const String &p_type) const{
        return ClassDB::is_parent_class(p_type, "cTorchModelData");
    }
    virtual String get_resource_type(const String &p_path) const{
        return "cTorchModelData";
    }
};

class cTorchModel : public Reference {
    GDCLASS(cTorchModel, Reference);
    private:
        torch::jit::script::Module module;
        cTorchModelData module_data;
        
    protected:
        static void _bind_methods(){
            ClassDB::bind_method(D_METHOD("init", "data"), &cTorchModel::init);
	        ClassDB::bind_method(D_METHOD("run", "input"), &cTorchModel::run);
            ClassDB::bind_method(D_METHOD("set_data", "data"), &cTorchModel::set_data);
	        ClassDB::bind_method(D_METHOD("get_data"), &cTorchModel::get_data);
            ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "model_data", PROPERTY_HINT_RESOURCE_TYPE, "cTorchModelData"), "set_data", "get_data");
        }
        void set_data(const Ref<cTorchModelData> &data){
            module_data.set_array(data->get_array());
            init(data);
        };
        Ref<cTorchModelData> get_data() const{
            return Ref<cTorchModelData>(&module_data);
        };

        void init(const Ref<cTorchModelData> &data){
            //Load model from pool byte array
            print_line("Loading from ModelData");
            char str[data->size()];
            data->save(&str[0]);
            std::stringstream strstream(std::string(str, data->size()));
            this->module = torch::jit::load(strstream);
        }

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