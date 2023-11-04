#include "vertex_data.hpp "

#include <limits>

#include <cstring> // for std::memcpy()

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/to_string.hpp"
#include "baked_model.hpp"

#include <iostream>

namespace
{

#	define ASSERTDIR_ "assets/cw2/"
	constexpr char const* kBakedObject = ASSERTDIR_ "sponza-pbr.comp5822mesh";
#	undef ASSERTDIR_

	BakedModel bakedModel = load_baked_model(kBakedObject);
}

void handle_baked_model(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, 
						std::vector<TextureFragment>& bakedfragments, std::vector<TextureFragment>& alphafragments)
{
	std::vector<TextureFragment> ret;

	for (int i = 0; i < bakedModel.meshes.size(); i++)
	{
		auto positions = bakedModel.meshes[i].positions;
		auto texcoords = bakedModel.meshes[i].texcoords;
		auto indices = bakedModel.meshes[i].indices;
		auto normals = bakedModel.meshes[i].normals;
		auto materialID = bakedModel.meshes[i].materialId;

		std::uint32_t baseColorTextureId = bakedModel.materials[materialID].baseColorTextureId;
		std::uint32_t roughnessTextureId = bakedModel.materials[materialID].roughnessTextureId;
		std::uint32_t metalnessTextureId = bakedModel.materials[materialID].metalnessTextureId;
		std::uint32_t alphaMaskTextureId = bakedModel.materials[materialID].alphaMaskTextureId;
		std::uint32_t idCount = indices.size();

		// Create final position and color buffers 
		lut::Buffer vertexPosGPU = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec3) * positions.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);
		lut::Buffer vertexColGPU = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec2) * texcoords.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);
		lut::Buffer vertexNmlGPU = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec3) * normals.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);
		lut::Buffer vertexIdsGPU = lut::create_buffer(
			aAllocator,
			sizeof(uint32_t) * indices.size(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		// Create the staging buffers
		lut::Buffer posStaging = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec3) * positions.size(),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);
		lut::Buffer colStaging = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec2) * texcoords.size(),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);
		lut::Buffer nmlStaging = lut::create_buffer(
			aAllocator,
			sizeof(glm::vec3) * texcoords.size(),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);
		lut::Buffer IdsStaging = lut::create_buffer(
			aAllocator,
			sizeof(uint32_t) * indices.size(),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		void* posPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, posStaging.allocation, &posPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());

		}
		std::memcpy(posPtr, positions.data(), sizeof(glm::vec3) * positions.size());
		vmaUnmapMemory(aAllocator.allocator, posStaging.allocation);

		void* colPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, colStaging.allocation, &colPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(colPtr, texcoords.data(), sizeof(glm::vec2) * texcoords.size());
		vmaUnmapMemory(aAllocator.allocator, colStaging.allocation);

		void* nmlPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, nmlStaging.allocation, &nmlPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(nmlPtr, normals.data(), sizeof(glm::vec3) * normals.size());
		vmaUnmapMemory(aAllocator.allocator, nmlStaging.allocation);

		void* IdsPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, IdsStaging.allocation, &IdsPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(IdsPtr, indices.data(), sizeof(uint32_t) * indices.size());
		vmaUnmapMemory(aAllocator.allocator, IdsStaging.allocation);


		lut::Fence uploadComplete = create_fence(aContext);

		// Queue data uploads from staging buffers to the final buffers 
		// This uses a separate command pool for simplicity. 
		lut::CommandPool uploadPool = create_command_pool(aContext);
		VkCommandBuffer uploadCmd = alloc_command_buffer(aContext, uploadPool.handle);

		// Record the copy commands into the command buffer
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo); VK_SUCCESS != res)
		{
			throw lut::Error("Beginning command buffer recording\n" "vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		VkBufferCopy pcopy{};
		pcopy.size = sizeof(glm::vec3) * positions.size();

		vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer, 1, &pcopy);

		lut::buffer_barrier(uploadCmd,
			vertexPosGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy ccopy{};
		ccopy.size = sizeof(glm::vec2) * texcoords.size();

		vkCmdCopyBuffer(uploadCmd, colStaging.buffer, vertexColGPU.buffer, 1, &ccopy);

		lut::buffer_barrier(uploadCmd,
			vertexColGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy ncopy{};
		ncopy.size = sizeof(glm::vec3) * normals.size();

		vkCmdCopyBuffer(uploadCmd, nmlStaging.buffer, vertexNmlGPU.buffer, 1, &ncopy);

		lut::buffer_barrier(uploadCmd,
			vertexNmlGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy icopy{};
		icopy.size = sizeof(uint32_t) * indices.size();

		vkCmdCopyBuffer(uploadCmd, IdsStaging.buffer, vertexIdsGPU.buffer, 1, &icopy);

		lut::buffer_barrier(uploadCmd,
			vertexIdsGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
		{
			throw lut::Error("Ending command buffer recording\n" "vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		// Submit transfer commands 
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &uploadCmd;

		if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo,
			uploadComplete.handle); VK_SUCCESS != res)
		{
			throw lut::Error("Submitting commands\n" "vkQueueSubmit() returned %s", lut::to_string(res).c_str());
		}

		if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle,
			VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Waiting for upload to complete\n" "vkWaitForFences() returned %s", lut::to_string(res).c_str());
		}
		if (alphaMaskTextureId == -1)
		{
			bakedfragments.emplace_back(TextureFragment{
			std::move(vertexPosGPU),
			std::move(vertexColGPU),
			std::move(vertexIdsGPU),
			std::move(vertexNmlGPU),
			baseColorTextureId,
			roughnessTextureId,
			metalnessTextureId,
			alphaMaskTextureId,
			idCount,
				});
		}
		else
		{
			alphafragments.emplace_back(TextureFragment{
			std::move(vertexPosGPU),
			std::move(vertexColGPU),
			std::move(vertexIdsGPU),
			std::move(vertexNmlGPU),
			baseColorTextureId,
			roughnessTextureId,
			metalnessTextureId,
			alphaMaskTextureId,
			idCount,
				});
			std::cout << bakedModel.textures[alphaMaskTextureId].path << std::endl;
		}
		
	}

	return;
}

void getImages(lut::VulkanWindow& window, VkCommandPool cpool_handle, lut::Allocator& allocator, std::vector<lut::Image>& imageVec, std::vector<lut::ImageView>& imageViewVec)
{
	int count = 0;

	for (auto& m_path : bakedModel.textures)
	{
		if (m_path.path == "") 
			throw lut::Error("failed to load texture");

		lut::Image tempImg;
		tempImg = lut::load_image_texture2d(m_path.path.c_str(), window, cpool_handle, allocator);
		
		lut::ImageView tempView;
		tempView = lut::create_image_view_texture2d(window, tempImg.image, VK_FORMAT_R8G8B8A8_SRGB);		
		
		imageVec.emplace_back(std::move(tempImg));
		imageViewVec.emplace_back(std::move(tempView));
		++count;
	}

	printf("load texture IMAGE: %d \n", count);
	return ;
}



