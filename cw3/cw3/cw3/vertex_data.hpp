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
#include "glm/glm.hpp"

namespace lut = labutils;


struct TextureFragment
{
	labutils::Buffer positions;
	labutils::Buffer texcoords;
	labutils::Buffer indices;
	labutils::Buffer normals;

	glm::vec3 baseColor;
	glm::vec3 emissiveColor;

	float roughness;
	float metalness;

	std::uint32_t idCount;
};

void getImages(lut::VulkanWindow& window, VkCommandPool cpool_handle, lut::Allocator& allocator, std::vector<lut::Image>& imageVec, std::vector<lut::ImageView>& imageViewVec);

std::vector<TextureFragment> handle_baked_model(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator);
//void handle_baked_model();
