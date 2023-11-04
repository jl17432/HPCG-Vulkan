#pragma once

#include <cstdint>

#include "../labutils/vulkan_context.hpp"

#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 

#include "simple_model.hpp"
#include "load_model_obj.hpp"
#include "vertex_data.hpp"

constexpr char const* kWaveObject = "assets/cw1/sponza_with_ship.obj";

struct ColorizedMesh
{
	labutils::Buffer positions;
	labutils::Buffer colors;
	std::vector<size_t> offsetVec;
	std::vector<size_t> vertexCount;
	std::size_t meshCount;
};

struct TexturedMesh 
{
	labutils::Buffer positions;
	labutils::Buffer texcoords;
	std::vector<std::uint32_t> texID;
	std::vector<size_t> offsetVec;
	std::vector<size_t> vertexCount;
	std::size_t meshCount;
};


TexturedMesh handle_textured_mesh(labutils::VulkanContext const&, labutils::Allocator const&);

ColorizedMesh handle_untextured_mesh(labutils::VulkanContext const&, labutils::Allocator const&);

std::vector<std::string> getTexPath();

SimpleModel const kWaveModel = load_simple_wavefront_obj(kWaveObject);





