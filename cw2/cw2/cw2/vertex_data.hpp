#pragma once

#include <cstdint>

#include "../labutils/vulkan_context.hpp"

#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 
#include "../labutils/vkimage.hpp"
#include "../labutils/vulkan_window.hpp"
#include "../labutils/vkobject.hpp"

#include <string.h>
#include <vector>

namespace lut = labutils;


struct ColorizedMesh
{
	labutils::Buffer positions;
	labutils::Buffer colors;

	std::uint32_t vertexCount;
};

struct TexturedMesh 
{ 
	labutils::Buffer positions; 
	labutils::Buffer texcoords; 

	std::uint32_t vertexCount; 
};

struct TextureFragment
{
	labutils::Buffer positions;
	labutils::Buffer texcoords;
	labutils::Buffer indices;
	labutils::Buffer normals;

	std::uint32_t baseColorTextureId;
	std::uint32_t roughnessTextureId;
	std::uint32_t metalnessTextureId;
	std::uint32_t alphaMaskTextureId = -1;
	std::uint32_t idCount;
};


void getImages(lut::VulkanWindow& window, VkCommandPool cpool_handle, lut::Allocator& allocator, std::vector<lut::Image>& imageVec, std::vector<lut::ImageView>& imageViewVec);

void handle_baked_model(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, std::vector<TextureFragment>& bakedfragments, std::vector<TextureFragment>& alphafragments);