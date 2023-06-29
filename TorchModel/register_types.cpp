#include "register_types.h"
#include "torch_model.h"
#include <gdextension_interface.h>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/classes/resource_loader.hpp>

using namespace godot;
static Ref<cTorchModelLoader> model_loader;

void initialize_cTorchModel_types(ModuleInitializationLevel p_level)
{
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	
	ClassDB::register_class<cTorchModelData>();
	ClassDB::register_class<cTorchModelLoader>();
	ClassDB::register_class<cTorchModel>();
	model_loader.instantiate();
    // ResourceLoader::add_resource_format_loader(model_loader);
	ResourceLoader::get_singleton()->add_resource_format_loader(model_loader, true);

	
}

void uninitialize_cTorchModel_types(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
    // ResourceLoader::remove_resource_format_loader(model_loader);
	ResourceLoader::get_singleton()->remove_resource_format_loader(model_loader);
	model_loader.unref();
}

extern "C"
{

	// Initialization.

	GDExtensionBool GDE_EXPORT torchmodel_library_init(const GDExtensionInterface *p_interface, const GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization)
	{
		GDExtensionBinding::InitObject init_obj(p_interface, p_library, r_initialization);

		init_obj.register_initializer(initialize_cTorchModel_types);
		init_obj.register_terminator(uninitialize_cTorchModel_types);
		init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

		return init_obj.init();
	}
}