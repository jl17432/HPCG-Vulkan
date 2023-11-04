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




struct TextureFragment
{
	labutils::Buffer positions;
	labutils::Buffer texcoords;
	labutils::Buffer indices;
	labutils::Buffer normals;
	labutils::Buffer tangents;

	std::uint32_t baseColorTextureId;
	std::uint32_t roughnessTextureId;
	std::uint32_t metalnessTextureId;
	std::uint32_t alphaMaskTextureId = -1;
	std::uint32_t normalMaskTextureId = -1;
	std::uint32_t idCount;
};


void getImages(lut::VulkanWindow& window, VkCommandPool cpool_handle, lut::Allocator& allocator, std::vector<lut::Image>& imageVec, std::vector<lut::ImageView>& imageViewVec);

void handle_baked_model(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, std::vector<TextureFragment>& bakedfragments, std::vector<TextureFragment>& alphafragments);
