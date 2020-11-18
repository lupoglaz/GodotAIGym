/* register_types.cpp */

#include "register_types.h"
#include "core/class_db.h"
#include "cSharedMemory.h"

static Ref<cTorchModelLoader> model_loader;

void register_GodotSharedMemory_types() {
	ClassDB::register_class<cSharedMemory>();
   ClassDB::register_class<cSharedMemorySemaphore>();
      
   ClassDB::register_class<cTorchModelData>();
   model_loader.instance();
   ResourceLoader::add_resource_format_loader(model_loader);

   ClassDB::register_class<cTorchModel>();
}

void unregister_GodotSharedMemory_types() {
   ResourceLoader::remove_resource_format_loader(model_loader);
   model_loader.unref();
}