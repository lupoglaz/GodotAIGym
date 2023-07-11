/* cSharedMemory.h */
#ifndef TORCH_MODEL_H
#define TORCH_MODEL_H

#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/templates/vector.hpp>
#include <godot_cpp/variant/builtin_types.hpp>
#include <godot_cpp/core/binder_common.hpp>
#include <godot_cpp/core/class_db.hpp>
#include "godot_cpp/variant/variant.hpp"
#include <godot_cpp/classes/resource_format_loader.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/godot.hpp>
using namespace godot;

#include <string>
#include <vector>
#include <exception>
#include <iostream>
#include <istream>
#include <streambuf>

#include <torch/script.h>
#include <torch/csrc/jit/serialization/export.h>
using namespace torch::indexing;

#include "godot_cpp/core/error_macros.hpp"

class cTorchModelData : public Resource{
    GDCLASS(cTorchModelData, Resource);

    private:
        PackedByteArray content;
        
    protected:
        static void _bind_methods(){
        	ClassDB::bind_method(D_METHOD("set_array", "array"), &cTorchModelData::set_array);
            ClassDB::bind_method(D_METHOD("get_array"), &cTorchModelData::get_array);
            ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "array"), "set_array", "get_array");
        }

    public:
        void load(const char *_data, size_t size){
            // print_line("Converting string to TypedArray<char>");
            //Convert model to poolbytearray
            content.resize(size);
            for(long i=0; i<size; i++)
                content[i] = _data[i];
        };
        long size() const{
            return content.size();
        }
        void save(char *_data) const{
            // print_line("Converting TypedArray<char> to string");
            for(int i=0; i<content.size(); i++)
                _data[i] = content[i];
        }

        void set_array(const PackedByteArray &p_array){
            content = p_array;
        };
        PackedByteArray get_array() const{
            return content;
        };
};

class cTorchModelLoader : public ResourceFormatLoader {
    GDCLASS(cTorchModelLoader, ResourceFormatLoader);
protected:
	static void _bind_methods() {}
public:
    virtual Variant _load(const String &p_path, const String &p_original_path, bool use_sub_threads, int32_t cache_mode) const{
        WARN_PRINT( (String("Loading from file to ModelData ") + p_path).ptr());
        //Load file in the resources to a model
        String glob_path;
        if(OS::get_singleton()->has_feature("editor")){
            glob_path = ProjectSettings::get_singleton()->globalize_path(p_path);
        }else{
            glob_path = OS::get_singleton()->get_executable_path().get_base_dir().path_join(p_path);
        }
        // WARN_PRINT( (String("Converted path to ") + glob_path).ptr() );
        
        torch::jit::script::Module module = torch::jit::load(glob_path.ascii().get_data());
        std::stringstream str;
        torch::jit::ExportModule(module, str);
        Ref<cTorchModelData> model_data = memnew(cTorchModelData);
        model_data->load(str.str().c_str(), str.str().size());
        
        return model_data;
    }
    virtual PackedStringArray _get_recognized_extensions() const{
        PackedStringArray psa;
	    psa.append("jit");
        // WARN_PRINT( (String("Recognized extensions ") + psa[0]).ptr() );
	    return psa;
    }
    virtual bool _handles_type(const String &p_type) const{
        // WARN_PRINT( (String("Handles resource type ") + p_type).ptr() );
        return p_type.to_lower().contains("torchmodel");
    }
    virtual String _get_resource_type(const String &p_path) const{
        // WARN_PRINT( (String("Get resource type ") + p_path).ptr() );
        String el = p_path.get_extension().to_lower();
	    if (el == "jit") {
            return "TorchModel";
        }
        return "";
        
    }
};

class cTorchModel : public RefCounted {
    GDCLASS(cTorchModel, RefCounted);
    private:
        torch::jit::script::Module module;
        cTorchModelData module_data;
        c10::InferenceMode guard;
        
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
            // print_line("Loading from ModelData");
            char str[data->size()];
            data->save(&str[0]);
            std::stringstream strstream(std::string(str, data->size()));
            this->module = torch::jit::load(strstream);
        }

        PackedFloat32Array run(PackedFloat32Array input){
            PackedFloat32Array output;
            // at::Tensor input_t = torch::zeros({1, input.size()}, torch::TensorOptions().dtype(torch::kFloat32));
            // for(int i=0; i<input.size(); i++) input_t.index_put_({0,i}, input[i]);
            at::Tensor input_t = torch::from_blob(input.ptrw(), {input.size()});
            // WARN_PRINT(String::num_int64(input.size()).ptr());
            // WARN_PRINT((String("") + String::num_int64(input_t.sizes()[0])).ptr());
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_t);
            at::Tensor output_t = this->module.forward(inputs).toTensor();
            auto output_t_a = output_t.accessor<float, 1>();
            for(int i=0; i<output_t.sizes()[0]; i++) output.push_back(output_t_a[i]);
            return output;
        }
};


#endif