#include "vertex_data.hpp"

#include <limits>
#include <iostream>

#include <cstring> // for std::memcpy()

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/to_string.hpp"
namespace lut = labutils;

std::vector<std::string> getTexPath()
{
	std::vector<std::string> res;

	for (auto material : kWaveModel.materials)
	{
		auto path = material.diffuseTexturePath;
		//if (path != "")
		//{
			res.emplace_back(material.diffuseTexturePath);
		//}
	}

	return res;
}


TexturedMesh handle_textured_mesh(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator)
{

	std::vector<std::uint32_t> texID;
	std::vector<size_t> offsetVec;
	std::vector<size_t> vertexCount;
	size_t meshCount = 0;

	for (auto& mesh : kWaveModel.meshes)
	{
		if (mesh.textured)
		{
			texID.emplace_back(mesh.materialIndex);
			offsetVec.emplace_back(mesh.vertexStartIndex);
			vertexCount.emplace_back(mesh.vertexCount);
			meshCount++;
		}
	}

	auto positions = kWaveModel.dataTextured.positions;
	auto texcoord = kWaveModel.dataTextured.texcoords;

	// Create final position and color buffers 
	lut::Buffer vertexPosGPU = lut::create_buffer(
		aAllocator,
		sizeof(glm::vec3) * positions.size(),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);
	lut::Buffer vertexColGPU = lut::create_buffer(
		aAllocator,
		sizeof(glm::vec2) * texcoord.size(),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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
		sizeof(glm::vec2) * texcoord.size(),
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

	std::memcpy(colPtr, texcoord.data(), sizeof(glm::vec2) * texcoord.size());
	vmaUnmapMemory(aAllocator.allocator, colStaging.allocation);

	// We need to ensure that the Vulkan resources are alive until all the 
	// transfers have completed. For simplicity, we will just wait for the 
	// operations to complete with a fence. A more complex solution might want 
	// to queue transfers, let these take place in the background while 
	// performing other tasks. 
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
	pcopy.size = sizeof(glm::vec3)*positions.size();

	vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer, 1, &pcopy);

	lut::buffer_barrier(uploadCmd,
		vertexPosGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	VkBufferCopy ccopy{};
	ccopy.size = sizeof(glm::vec2)*texcoord.size();

	vkCmdCopyBuffer(uploadCmd, colStaging.buffer, vertexColGPU.buffer, 1, &ccopy);

	lut::buffer_barrier(uploadCmd,
		vertexColGPU.buffer,
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

	// Wait for commands to finish before we destroy the temporary resources 
	// required for the transfers (staging buffers, command pool, ...) 
	// 
	// The code doesn’t destory the resources implicitly – the resources are 
	// destroyed by the destructors of the labutils wrappers for the various 
	// objects once we leave the function’s scope. 
	if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle,
		VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
	{
		throw lut::Error("Waiting for upload to complete\n" "vkWaitForFences() returned %s", lut::to_string(res).c_str());
	}

	unsigned int numElements = static_cast<unsigned int>(positions.size());
	return TexturedMesh{
		std::move(vertexPosGPU),
		std::move(vertexColGPU),
		texID,
		offsetVec,
		vertexCount,  // now three floats per position 
		meshCount
	};
}



ColorizedMesh handle_untextured_mesh(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator)
{
	std::vector<glm::vec3> positions = kWaveModel.dataUntextured.positions;
	std::vector<glm::vec3> colours;
	std::vector<size_t> offsetVec;
	std::vector<size_t> vertexCount;
	std::size_t meshCount = 0;

	for (auto& mesh : kWaveModel.meshes)
	{
		if (!mesh.textured)
		{
			auto colour = kWaveModel.materials[mesh.materialIndex].diffuseColor;
			for (auto i = 0; i < mesh.vertexCount; i++)
			{
				colours.emplace_back(colour);
			}
			offsetVec.emplace_back(mesh.vertexStartIndex);
			vertexCount.emplace_back(mesh.vertexCount);
			meshCount++;
		}		
	}

	// Create final position and color buffers 
	lut::Buffer vertexPosGPU = lut::create_buffer(
		aAllocator,
		sizeof(glm::vec3) * positions.size(),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);
	lut::Buffer vertexColGPU = lut::create_buffer(
		aAllocator,
		sizeof(glm::vec3) * colours.size(),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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
		sizeof(glm::vec3) * colours.size(),
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

	std::memcpy(colPtr, colours.data(), sizeof(glm::vec3) * colours.size());
	vmaUnmapMemory(aAllocator.allocator, colStaging.allocation);

	// We need to ensure that the Vulkan resources are alive until all the 
	// transfers have completed. For simplicity, we will just wait for the 
	// operations to complete with a fence. A more complex solution might want 
	// to queue transfers, let these take place in the background while 
	// performing other tasks. 
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
	ccopy.size = sizeof(glm::vec3) * colours.size();
	vkCmdCopyBuffer(uploadCmd, colStaging.buffer, vertexColGPU.buffer, 1, &ccopy);

	lut::buffer_barrier(uploadCmd,
		vertexColGPU.buffer,
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

	// Wait for commands to finish before we destroy the temporary resources 
	// required for the transfers (staging buffers, command pool, ...) 
	// 
	// The code doesn’t destory the resources implicitly – the resources are 
	// destroyed by the destructors of the labutils wrappers for the various 
	// objects once we leave the function’s scope. 
	if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle,
		VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
	{
		throw lut::Error("Waiting for upload to complete\n" "vkWaitForFences() returned %s", lut::to_string(res).c_str());
	}
	return ColorizedMesh{
		std::move(vertexPosGPU),
		std::move(vertexColGPU),
		offsetVec,
		vertexCount, // now three floats per position 
		meshCount
	};

}