#include "glm/fwd.hpp"
#include <volk/volk.h>

#include <tuple>
#include <chrono>
#include <limits>
#include <vector>
#include <stdexcept>
#include <unordered_map>
#include <array>

#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <volk/volk.h>

#if !defined(GLM_FORCE_RADIANS)
#	define GLM_FORCE_RADIANS
#endif
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../labutils/to_string.hpp"
#include "../labutils/vulkan_window.hpp"

#include "../labutils/angle.hpp"
using namespace labutils::literals;

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkimage.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 
namespace lut = labutils;

#include "baked_model.hpp"
#include "vertex_data.hpp"


namespace
{
	using Clock_ = std::chrono::steady_clock;
	using Secondsf_ = std::chrono::duration<float, std::ratio<1>>;

	namespace cfg
	{

		// Compiled shader code for the graphics pipeline
		// See sources in exercise4/shaders/*. 
#		define SHADERDIR_ "assets/cw3/shaders/"
		constexpr char const* v_DefaultShaderPath = SHADERDIR_ "lightTex.vert.spv";
		constexpr char const* f_DefaultShaderPath = SHADERDIR_ "lightTex.frag.spv";

		constexpr char const* v_ComputeShaderPath = SHADERDIR_ "default.vert.spv";
		constexpr char const* f_ComputeShaderPath = SHADERDIR_ "default.frag.spv";

		constexpr char const* v_pbrShaderPath = SHADERDIR_ "pbr.vert.spv"; 
		constexpr char const* f_pbrShaderPath = SHADERDIR_ "pbr.frag.spv"; 
				
#		undef SHADERDIR_

		constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

		constexpr float kCameraNear = 0.1f;
		constexpr float kCameraFar = 100.f;

		constexpr auto kCameraFov = 60.0_degf;

		constexpr float kCameraBaseSpeed = 1.7f; // units/second 
		constexpr float kCameraFastMult = 5.f; // speed multiplier 
		constexpr float kCameraSlowMult = 0.05f; // speed multiplier 

		constexpr float kCameraMouseSensitivity = 0.01f; // radians per pixel

		constexpr float kLightRotatingSpeed = 0.01f;
		constexpr float kLightMovingSpeed = 2.f;

	}


	struct PushConstantData {
		float baseColor_r;
		float baseColor_g;
		float baseColor_b;

		float emissiveColor_r;
		float emissiveColor_g;
		float emissiveColor_b;

		float roughness;
		float metalness;
	};

	struct PushBoundaryData {
		float width;
		float height;
	};


	// for keyboard control
	enum class EInputState
	{
		////////////////////////////////////////////
		//										  //
		//		     Camera Movement			  //
		//										  //
		////////////////////////////////////////////

		forward,
		backward,
		strafeLeft,
		strafeRight,
		levitate,
		sink,
		fast,
		slow,
		mousing,


		////////////////////////////////////////////
		//										  //
		//		      light Movement			  //
		//										  //
		////////////////////////////////////////////

		lightRotate_Y,
		lightForward,
		lightBackward,
		lightLeft,
		lightRight,
		lightUp,
		lightDown,

		max,
	}; // end of EInputState

	// for updating the camera state
	struct UserState
	{
		bool inputMap[std::size_t(EInputState::max)] = {};

		float mouseX = 0.f, mouseY = 0.f;
		float previousX = 0.f, previousY = 0.f;
		glm::vec3 lightPosWorld = glm::vec3(0.f, 2.f, 0.f);

		bool wasMousing = false;
		bool lightRotate = false;

		glm::mat4 camera2world = glm::identity<glm::mat4>();

	};

	void update_user_state(UserState&, float aElapsedTime);


	// GLFW callbacks
	void glfw_callback_key_press(GLFWwindow*, int, int, int, int);
	void glfw_callback_button(GLFWwindow*, int, int, int);
	void glfw_callback_motion(GLFWwindow*, double, double);

	namespace glsl
	{
		struct SceneUniform
		{
			// Note: need to be careful about the packing/alignment here! 

			glm::mat4 camera;
			glm::mat4 projection;
			glm::mat4 projCam;
			glm::vec3 lightPos;
			glm::vec3 lightCol;
		};

		static_assert(sizeof(SceneUniform) <= 65536, "SceneUniform must be less than 65536 bytes for vkCmdUpdateBuffer");
		static_assert(sizeof(SceneUniform) % 4 == 0, "SceneUniform size must be a multiple of 4 bytes");
	}

	// Helpers:

	lut::RenderPass create_render_pass(lut::VulkanWindow const&);
	lut::RenderPass create_compute_pass(lut::VulkanWindow const& aWindow);
	lut::RenderPass create_pbr_pass(lut::VulkanWindow const& aWindow);


	lut::DescriptorSetLayout create_scene_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_object_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_compute_descriptor_layout(lut::VulkanWindow const& aWindow);
	lut::DescriptorSetLayout create_pbr_descriptor_layout(lut::VulkanWindow const& aWindow);

	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const&, VkDescriptorSetLayout const&, VkDescriptorSetLayout const&);
	lut::PipelineLayout create_compute_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout const& aSceneLayout, VkDescriptorSetLayout const aObjectLayout);
	lut::PipelineLayout create_pbr_pipeline_layout(lut::VulkanContext const&, VkDescriptorSetLayout const&, VkDescriptorSetLayout const&);

	lut::Pipeline create_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);
	lut::Pipeline create_compute_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout);
	lut::Pipeline create_pbr_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);


	std::tuple<lut::Image, lut::ImageView> create_empty_image_view(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator, VkFormat format);

	void create_compute_framebuffer(
		lut::VulkanWindow const& aWindow,
		VkRenderPass aRenderPass,
		lut::Framebuffer& aFramebuffers,
		VkImageView aDepthView,
		VkImageView emptyView,
		VkImageView positionView,
		VkImageView normalView,
		VkImageView materialView,
		VkImageView emissiveView
	);

	void create_pbr_framebuffer(
		lut::VulkanWindow const& aWindow,
		VkRenderPass aRenderPass,
		lut::Framebuffer& aFramebuffers,
		VkImageView pbrView,
		VkImageView filterView
	);

	void create_render_descriptorSet(
		lut::VulkanWindow const& aWindow,
		VkDescriptorSet const& renderDescriptorSet,
		VkSampler const& sampler,
		VkImageView const& pbrImageView,
		VkImageView const& filterImageView);

	void create_pbr_descriptorSet(
		lut::VulkanWindow const& aWindow,
		VkDescriptorSet const& renderDescriptorSet,
		VkImageView const& baseImageView,
		VkSampler const& sampler,
		VkImageView const& positionImageView,
		VkImageView const& normalImageView,
		VkImageView const& materialImageView,
		VkImageView const& emissiveImageView);

	void create_swapchain_framebuffers(
		lut::VulkanWindow const&,
		VkRenderPass,
		std::vector<lut::Framebuffer>&,
		VkImageView aDepthView
	);

	void update_scene_uniforms(
		glsl::SceneUniform&,
		std::uint32_t aFramebufferWidth,
		std::uint32_t aFramebufferHeight,
		UserState const& aState
	);


	void record_commands(VkCommandBuffer aCmdBuff,
		VkRenderPass aRenderPass,
		VkFramebuffer aFramebuffer,
		VkPipeline aGraphicsPipe,
		VkExtent2D const& aImageExtent,
		VkBuffer aSceneUBO,
		glsl::SceneUniform& aSceneUniform,
		VkPipelineLayout aGraphicsLayout,
		VkDescriptorSet aSceneDescriptors,
		VkDescriptorSet& myDescriptor,
		std::vector<TextureFragment>& bakedfragments,

		VkRenderPass compute_renderpass,
		VkFramebuffer compute_framebuffer,
		VkPipeline computePipe,
		VkPipelineLayout compute_pipeline_layout,

		VkRenderPass pbr_renderpass,
		VkFramebuffer pbr_framebuffer,
		VkPipeline pbrPipe,
		VkPipelineLayout pbr_pipeline_layout,
		VkDescriptorSet& pbr_descriptor);

	void submit_commands(
		lut::VulkanWindow const&,
		VkCommandBuffer,
		VkFence,
		VkSemaphore,
		VkSemaphore
	);
	void present_results(
		VkQueue,
		VkSwapchainKHR,
		std::uint32_t aImageIndex,
		VkSemaphore,
		bool& aNeedToRecreateSwapchain
	);


	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const&, lut::Allocator const&);

	// Local types/structures:

	// Local functions:
}

int main() try
{
	// Create Vulkan Window
	auto window = lut::make_vulkan_window();

	// Configure the GLFW window 
	UserState state{};
	glfwSetWindowUserPointer(window.window, &state);

	// Configure the GLFW window
	glfwSetKeyCallback(window.window, &glfw_callback_key_press);
	glfwSetMouseButtonCallback(window.window, &glfw_callback_button);
	glfwSetCursorPosCallback(window.window, &glfw_callback_motion);

	// Create VMA allocator
	lut::Allocator allocator = lut::create_allocator(window);

	// Intialize resources
	lut::RenderPass renderPass = create_render_pass(window);
	lut::RenderPass compute_renderpass = create_compute_pass(window);
	lut::RenderPass pbr_renderpass = create_pbr_pass(window);


	lut::DescriptorSetLayout sceneLayout = create_scene_descriptor_layout(window);
	lut::DescriptorSetLayout objectLayout = create_object_descriptor_layout(window);
	lut::DescriptorSetLayout computeLayout = create_compute_descriptor_layout(window);
	lut::DescriptorSetLayout pbrLayout = create_pbr_descriptor_layout(window);




	lut::PipelineLayout pipeLayout = create_pipeline_layout(window, sceneLayout.handle, objectLayout.handle);
	lut::Pipeline pipe = create_pipeline(window, renderPass.handle, pipeLayout.handle);

	lut::PipelineLayout compute_pipeLayout = create_compute_pipeline_layout(window, sceneLayout.handle, computeLayout.handle);
	lut::Pipeline compute_pipe = create_compute_pipeline(window, compute_renderpass.handle, compute_pipeLayout.handle);

	lut::PipelineLayout pbr_pipeLayout = create_pbr_pipeline_layout(window, sceneLayout.handle, pbrLayout.handle);
	lut::Pipeline pbr_pipe = create_pbr_pipeline(window, pbr_renderpass.handle, pbr_pipeLayout.handle);


	auto [depthBuffer, depthBufferView] = create_depth_buffer(window, allocator);
	auto [baseImage, baseImageView] = create_empty_image_view(window, allocator, VK_FORMAT_B8G8R8A8_SRGB);
	auto [positionImage, positionImageView] = create_empty_image_view(window, allocator, VK_FORMAT_R32G32B32A32_SFLOAT);
	auto [normalImage, normalImageView] = create_empty_image_view(window, allocator, VK_FORMAT_R32G32B32A32_SFLOAT);
	auto [materialImage, materialImageView] = create_empty_image_view(window, allocator, VK_FORMAT_B8G8R8A8_SRGB);
	auto [emissiveImage, emissiveImageView] = create_empty_image_view(window, allocator, VK_FORMAT_B8G8R8A8_SRGB);


	auto [pbrImage, pbrImageView] = create_empty_image_view(window, allocator, VK_FORMAT_B8G8R8A8_SRGB);
	auto [filterImage, filterImageView] = create_empty_image_view(window, allocator, VK_FORMAT_B8G8R8A8_SRGB);


	std::vector<lut::Framebuffer> framebuffers;
	create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBufferView.handle);

	lut::CommandPool cpool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	std::vector<VkCommandBuffer> cbuffers;
	std::vector<lut::Fence> cbfences;

	for (std::size_t i = 0; i < framebuffers.size(); ++i)
	{
		cbuffers.emplace_back(lut::alloc_command_buffer(window, cpool.handle));
		cbfences.emplace_back(lut::create_fence(window, VK_FENCE_CREATE_SIGNALED_BIT));
	}

	lut::Semaphore imageAvailable = lut::create_semaphore(window);
	lut::Semaphore renderFinished = lut::create_semaphore(window);

	// Load data
	std::vector<TextureFragment> bakedfragments = handle_baked_model(window, allocator);

	//TODO- (Section 3) create scene uniform buffer with lut::create_buffer()
	lut::Buffer sceneUBO = lut::create_buffer(
		allocator,
		sizeof(glsl::SceneUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	lut::Framebuffer compute_framebuffer;
	create_compute_framebuffer(
		window, 
		compute_renderpass.handle, 
		compute_framebuffer,
		depthBufferView.handle, 
		baseImageView.handle,
		positionImageView.handle, 
		normalImageView.handle,
		materialImageView.handle,
		emissiveImageView.handle);

	lut::Framebuffer pbr_framebuffer;
	create_pbr_framebuffer(
		window,
		pbr_renderpass.handle,
		pbr_framebuffer,
		pbrImageView.handle,
		filterImageView.handle
	);




	//TODO- (Section 3) create descriptor pool
	lut::DescriptorPool dpool = lut::create_descriptor_pool(window);

	//TODO- (Section 3) allocate descriptor set for uniform buffer
	VkDescriptorSet sceneDescriptors = lut::alloc_desc_set(window, dpool.handle, sceneLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorBufferInfo sceneUboInfo{};
		sceneUboInfo.buffer = sceneUBO.buffer;
		sceneUboInfo.range = VK_WHOLE_SIZE;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = sceneDescriptors;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &sceneUboInfo;


		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}
	lut::Sampler defaultSampler = lut::create_default_sampler(window);



	///////////////////////////////////////////
	//										 //
	//      Creating object descriptors	     //
	//										 //
	///////////////////////////////////////////


	VkDescriptorSet render_descriptor = lut::alloc_desc_set(window, dpool.handle, objectLayout.handle);
	create_render_descriptorSet(window, 
		render_descriptor, 
		defaultSampler.handle, 
		pbrImageView.handle,
		filterImageView.handle);

	VkDescriptorSet pbr_descriptor = lut::alloc_desc_set(window, dpool.handle, pbrLayout.handle);
	create_pbr_descriptorSet(window,
		pbr_descriptor,
		baseImageView.handle,
		defaultSampler.handle,
		positionImageView.handle,
		normalImageView.handle,
		materialImageView.handle,
		emissiveImageView.handle);
	
	



	// Application main loop
	bool recreateSwapchain = false;

	// record the current time
	auto previousClock = Clock_::now();

	while (!glfwWindowShouldClose(window.window))
	{
		// Let GLFW process events.
		// glfwPollEvents() checks for events, processes them. If there are no
		// events, it will return immediately. Alternatively, glfwWaitEvents()
		// will wait for any event to occur, process it, and only return at
		// that point. The former is useful for applications where you want to
		// render as fast as possible, whereas the latter is useful for
		// input-driven applications, where redrawing is only needed in
		// reaction to user input (or similar).
		glfwPollEvents(); // or: glfwWaitEvents()

		// Recreate swap chain?
		if (recreateSwapchain)
		{
			// We need to destroy several objects, which may still be in use by 
			// the GPU. Therefore, first wait for the GPU to finish processing. 
			vkDeviceWaitIdle(window.device);

			// Recreate them 
			auto const changes = recreate_swapchain(window);

			if (changes.changedFormat)
			{
				renderPass = create_render_pass(window);
				compute_renderpass = create_compute_pass(window);
				pbr_renderpass = create_pbr_pass(window);
			}
				
				
			if (changes.changedSize)
			{
				std::tie(depthBuffer, depthBufferView) = create_depth_buffer(window, allocator);

				std::tie(baseImage, baseImageView) = create_empty_image_view(window, allocator, VK_FORMAT_B8G8R8A8_SRGB);
				std::tie(positionImage, positionImageView) = create_empty_image_view(window, allocator, VK_FORMAT_R32G32B32A32_SFLOAT);
				std::tie(normalImage, normalImageView) = create_empty_image_view(window, allocator, VK_FORMAT_R32G32B32A32_SFLOAT);
				std::tie(materialImage, materialImageView) = create_empty_image_view(window, allocator, VK_FORMAT_B8G8R8A8_SRGB);
				std::tie(emissiveImage, emissiveImageView) = create_empty_image_view(window, allocator, VK_FORMAT_B8G8R8A8_SRGB);

				std::tie(pbrImage, pbrImageView) = create_empty_image_view(window, allocator, VK_FORMAT_B8G8R8A8_SRGB);
				std::tie(filterImage, filterImageView) = create_empty_image_view(window, allocator, VK_FORMAT_B8G8R8A8_SRGB);
			}

			framebuffers.clear();
			create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBufferView.handle);

			create_compute_framebuffer(window, 
				compute_renderpass.handle, 
				compute_framebuffer, 
				depthBufferView.handle, 
				baseImageView.handle, 
				positionImageView.handle,
				normalImageView.handle,
				materialImageView.handle,
				emissiveImageView.handle);

			create_pbr_framebuffer(window,
				pbr_renderpass.handle,
				pbr_framebuffer,
				pbrImageView.handle,
				filterImageView.handle);



			create_render_descriptorSet(window,
				render_descriptor,
				defaultSampler.handle,
				pbrImageView.handle,
				filterImageView.handle);

			create_pbr_descriptorSet(window,
				pbr_descriptor,
				baseImageView.handle,
				defaultSampler.handle,
				positionImageView.handle,
				normalImageView.handle,
				materialImageView.handle,
				emissiveImageView.handle
				);

			if (changes.changedSize)
			{
				pipe = create_pipeline(window, renderPass.handle, pipeLayout.handle);
				compute_pipe = create_compute_pipeline(window, compute_renderpass.handle, compute_pipeLayout.handle);
				pbr_pipe = create_pbr_pipeline(window, pbr_renderpass.handle, pbr_pipeLayout.handle);
			}

			recreateSwapchain = false;
			continue;
		}

		//TODO- (Section 1) Exercise 3:

		std::uint32_t imageIndex = 0;
		auto const acquireRes = vkAcquireNextImageKHR(
			window.device,
			window.swapchain,
			std::numeric_limits<std::uint64_t>::max(),
			imageAvailable.handle,
			VK_NULL_HANDLE,
			&imageIndex
		);

		if (VK_SUBOPTIMAL_KHR == acquireRes || VK_ERROR_OUT_OF_DATE_KHR == acquireRes)
		{
			recreateSwapchain = true;
			continue;
		}

		if (VK_SUCCESS != acquireRes)
		{
			throw lut::Error("Unable to acquire next swapchain image\n" "vkAcquireNextImageKHR() returned %s", lut::to_string(acquireRes).c_str());
		}

		//TODO- (Section 1)  - wait for command buffer to be available
		assert(std::size_t(imageIndex) < cbfences.size());

		if (auto const res = vkWaitForFences(window.device, 1, &cbfences[imageIndex].handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to wait for command buffer fence %u\n" "vkWaitForFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}

		if (auto const res = vkResetFences(window.device, 1, &cbfences[imageIndex].handle); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to reset command buffer fence %u\n" "vkResetFences() returned %s", imageIndex, lut::to_string(res).c_str());

		}

		// Update state 
		auto const now = Clock_::now();
		auto const dt = std::chrono::duration_cast<Secondsf_>(now - previousClock).count();
		previousClock = now;
		update_user_state(state, dt);

		// Prepare data for this frame 
		glsl::SceneUniform sceneUniforms{};
		update_scene_uniforms(sceneUniforms, window.swapchainExtent.width, window.swapchainExtent.height, state);

		//TODO- (Section 1)  - record and submit commands
		// Record and submit commands for this frame 
		assert(std::size_t(imageIndex) < cbuffers.size());
		assert(std::size_t(imageIndex) < framebuffers.size());

		record_commands(
			cbuffers[imageIndex],
			renderPass.handle,
			framebuffers[imageIndex].handle,
			pipe.handle,
			window.swapchainExtent,
			sceneUBO.buffer,
			sceneUniforms,
			pipeLayout.handle,
			sceneDescriptors,
			render_descriptor,
			bakedfragments,

			compute_renderpass.handle,
			compute_framebuffer.handle,
			compute_pipe.handle,
			compute_pipeLayout.handle,

			pbr_renderpass.handle,
			pbr_framebuffer.handle,
			pbr_pipe.handle,
			pbr_pipeLayout.handle,
			pbr_descriptor
		);

		submit_commands(
			window,
			cbuffers[imageIndex],
			cbfences[imageIndex].handle,
			imageAvailable.handle,
			renderFinished.handle
		);


		//TODO- (Section 1)  - present rendered images (note: use the present_results() method)
		// Present the results 

		present_results(window.presentQueue, window.swapchain, imageIndex, renderFinished.handle, recreateSwapchain);

	}

	// Cleanup takes place automatically in the destructors, but we sill need
	// to ensure that all Vulkan commands have finished before that.
	vkDeviceWaitIdle(window.device);

	return 0;
}
catch (std::exception const& eErr)
{
	std::fprintf(stderr, "\n");
	std::fprintf(stderr, "Error: %s\n", eErr.what());
	return 1;
}

namespace
{
	void glfw_callback_key_press(GLFWwindow* aWindow, int aKey, int /*aScanCode*/, int aAction, int /*aModifierFlags*/)
	{
		if (GLFW_KEY_ESCAPE == aKey && GLFW_PRESS == aAction)
		{
			glfwSetWindowShouldClose(aWindow, GLFW_TRUE);
		}

		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWindow));
		assert(state);

		bool const isReleased = (GLFW_RELEASE == aAction);


		////////////////////////////////////////////
		//										  //
		//		     Camera Movement			  //
		//										  //
		////////////////////////////////////////////

		switch (aKey)
		{
		case GLFW_KEY_W:
			state->inputMap[std::size_t(EInputState::forward)] = !isReleased;
			break;
		case GLFW_KEY_S:
			state->inputMap[std::size_t(EInputState::backward)] = !isReleased;
			break;
		case GLFW_KEY_A:
			state->inputMap[std::size_t(EInputState::strafeLeft)] = !isReleased;
			break;
		case GLFW_KEY_D:
			state->inputMap[std::size_t(EInputState::strafeRight)] = !isReleased;
			break;
		case GLFW_KEY_E:
			state->inputMap[std::size_t(EInputState::levitate)] = !isReleased;
			break;
		case GLFW_KEY_Q:
			state->inputMap[std::size_t(EInputState::sink)] = !isReleased;
			break;

		case GLFW_KEY_LEFT_SHIFT: [[fallthrough]];
		case GLFW_KEY_RIGHT_SHIFT:
			state->inputMap[std::size_t(EInputState::fast)] = !isReleased;
			break;

		case GLFW_KEY_LEFT_CONTROL: [[fallthrough]];
		case GLFW_KEY_RIGHT_CONTROL:
			state->inputMap[std::size_t(EInputState::slow)] = !isReleased;
			break;



			////////////////////////////////////////////
			//										  //
			//		      light Movement			  //
			//										  //
			////////////////////////////////////////////

		case GLFW_KEY_SPACE:
			state->inputMap[std::size_t(EInputState::lightRotate_Y)] = !isReleased;
			break;
		case GLFW_KEY_I:
			state->inputMap[std::size_t(EInputState::lightForward)] = !isReleased;
			break;
		case GLFW_KEY_J:
			state->inputMap[std::size_t(EInputState::lightLeft)] = !isReleased;
			break;
		case GLFW_KEY_K:
			state->inputMap[std::size_t(EInputState::lightBackward)] = !isReleased;
			break;
		case GLFW_KEY_L:
			state->inputMap[std::size_t(EInputState::lightRight)] = !isReleased;
			break;
		case GLFW_KEY_U:
			state->inputMap[std::size_t(EInputState::lightUp)] = !isReleased;
			break;
		case GLFW_KEY_O:
			state->inputMap[std::size_t(EInputState::lightDown)] = !isReleased;
			break;


		default:
			;
		}
	}

	void glfw_callback_button(GLFWwindow* aWin, int aBut, int aAct, int)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		if (GLFW_MOUSE_BUTTON_RIGHT == aBut && GLFW_PRESS == aAct)
		{
			auto& flag = state->inputMap[std::size_t(EInputState::mousing)];

			flag = !flag;
			if (flag)
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			else
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}

	void glfw_callback_motion(GLFWwindow* aWin, double aX, double aY)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		state->mouseX = float(aX);
		state->mouseY = float(aY);
	}

	void update_user_state(UserState& aState, float aElapsedTime)
	{
		auto& cam = aState.camera2world;
		auto& light = aState.lightPosWorld;


		////////////////////////////////////////////
		//										  //
		//		     Camera Movement			  //
		//										  //
		////////////////////////////////////////////

		if (aState.inputMap[std::size_t(EInputState::mousing)])
		{
			// Only update the rotation on the second frame of mouse 
			// navigation. This ensures that the previousX and Y variables are 
			// initialized to sensible values. 
			if (aState.wasMousing)
			{
				auto const sens = cfg::kCameraMouseSensitivity;
				auto const dx = sens * (aState.mouseX - aState.previousX);
				auto const dy = sens * (aState.mouseY - aState.previousY);

				cam = cam * glm::rotate(-dy, glm::vec3(1.f, 0.f, 0.f));
				cam = cam * glm::rotate(-dx, glm::vec3(0.f, 1.f, 0.f));
			}

			aState.previousX = aState.mouseX;
			aState.previousY = aState.mouseY;
			aState.wasMousing = true;
		}
		else
		{
			aState.wasMousing = false;
		}


		auto const move = aElapsedTime * cfg::kCameraBaseSpeed *
			(aState.inputMap[std::size_t(EInputState::fast)] ? cfg::kCameraFastMult : 1.f) *
			(aState.inputMap[std::size_t(EInputState::slow)] ? cfg::kCameraSlowMult : 1.f)
			;

		if (aState.inputMap[std::size_t(EInputState::forward)])
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, -move));
		if (aState.inputMap[std::size_t(EInputState::backward)])
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, +move));

		if (aState.inputMap[std::size_t(EInputState::strafeLeft)])
			cam = cam * glm::translate(glm::vec3(-move, 0.f, 0.f));
		if (aState.inputMap[std::size_t(EInputState::strafeRight)])
			cam = cam * glm::translate(glm::vec3(+move, 0.f, 0.f));

		if (aState.inputMap[std::size_t(EInputState::levitate)])
			cam = cam * glm::translate(glm::vec3(0.f, +move, 0.f));
		if (aState.inputMap[std::size_t(EInputState::sink)])
			cam = cam * glm::translate(glm::vec3(0.f, -move, 0.f));




		////////////////////////////////////////////
		//										  //
		//		      light Movement			  //
		//										  //
		////////////////////////////////////////////

		if (aState.inputMap[std::size_t(EInputState::lightRotate_Y)])
		{
			if (aState.lightRotate)
			{
				auto const speed = cfg::kLightRotatingSpeed;
				light = glm::rotate(speed, glm::vec3(0.f, 1.f, 0.f)) * glm::vec4(light, 1.f);
			}

			aState.lightRotate = true;
		}
		else
		{
			aState.lightRotate = false;
		}

		auto const lightMove = aElapsedTime * cfg::kLightMovingSpeed;

		if (aState.inputMap[std::size_t(EInputState::lightForward)])
			light = glm::translate(glm::vec3(0.f, 0.f, lightMove)) * glm::vec4(light, 1.f);
		if (aState.inputMap[std::size_t(EInputState::lightBackward)])
			light = glm::translate(glm::vec3(0.f, 0.f, -lightMove)) * glm::vec4(light, 1.f);
		if (aState.inputMap[std::size_t(EInputState::lightLeft)])
			light = glm::translate(glm::vec3(lightMove, 0.f, 0.f)) * glm::vec4(light, 1.f);
		if (aState.inputMap[std::size_t(EInputState::lightRight)])
			light = glm::translate(glm::vec3(-lightMove, 0.f, 0.f)) * glm::vec4(light, 1.f);
		if (aState.inputMap[std::size_t(EInputState::lightUp)])
			light = glm::translate(glm::vec3(0.f, lightMove, 0.f)) * glm::vec4(light, 1.f);
		if (aState.inputMap[std::size_t(EInputState::lightDown)])
			light = glm::translate(glm::vec3(0.f, -lightMove, 0.f)) * glm::vec4(light, 1.f);

	}

}

namespace
{
	void update_scene_uniforms(glsl::SceneUniform& aSceneUniforms, std::uint32_t aFramebufferWidth, std::uint32_t aFramebufferHeight, UserState const& aState)
	{
		//TODO- (Section 3) initialize SceneUniform members
		float const aspect = aFramebufferWidth / float(aFramebufferHeight);

		aSceneUniforms.projection = glm::perspectiveRH_ZO(
			lut::Radians(cfg::kCameraFov).value(),
			aspect,
			cfg::kCameraNear,
			cfg::kCameraFar
		);
		aSceneUniforms.projection[1][1] *= -1.f; // mirror Y axis 

		//aSceneUniforms.camera = glm::translate(glm::vec3(0.f, -0.3f, -1.f));
		aSceneUniforms.camera = glm::inverse(aState.camera2world);

		aSceneUniforms.projCam = aSceneUniforms.projection * aSceneUniforms.camera;

		aSceneUniforms.lightPos = aState.lightPosWorld;
		aSceneUniforms.lightCol = glm::vec3(1.f, 1.f, 1.f);

	}
}

namespace
{
	lut::RenderPass create_render_pass(lut::VulkanWindow const& aWindow)
	{
		VkAttachmentDescription attachments[1]{};
		attachments[0].format = aWindow.swapchainFormat; //changed! 
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; //changed! 

		VkAttachmentReference subpassAttachments[1]{};
		subpassAttachments[0].attachment = 0; // this refers to attachments[0] 
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = subpassAttachments;

		VkSubpassDependency dep1 = {};
		dep1.srcSubpass = VK_SUBPASS_EXTERNAL;
		dep1.dstSubpass = 0;
		dep1.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dep1.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dep1.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dep1.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dep1.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkSubpassDependency dep2 = {};
		dep2.srcSubpass = 0;
		dep2.dstSubpass = VK_SUBPASS_EXTERNAL;
		dep2.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dep2.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dep2.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dep2.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dep2.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkSubpassDependency dependencies[2] = { dep1, dep2 };


		// changed: no explicit subpass dependencies 

		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 1;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 0; //changed! 
		passInfo.pDependencies = nullptr; //changed! 

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass
		); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create render pass\n" "vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);
	}

	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout const& aSceneLayout, VkDescriptorSetLayout const& aObjectLayout)
	{

		VkDescriptorSetLayout layouts[] = {
			// Order must match the set = N in the shaders 
			aSceneLayout,
			aObjectLayout
		};

		VkPushConstantRange pushConstantRange{};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT; // Or the stage where the float value will be used
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(PushBoundaryData);

		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]); // updated!
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 1;
		layoutInfo.pPushConstantRanges = &pushConstantRange;


		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr,
			&layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n" "vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
		}
		return lut::PipelineLayout(aContext.device, layout);
	}

	lut::Pipeline create_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::v_DefaultShaderPath);;
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::f_DefaultShaderPath);

		// Define shader stages in the pipeline 
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";




		//====================================================================================================================================


		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		

		// Define viewport and scissor regions 
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0, 0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width, aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// Define rasterization options 
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.f; // required.

		// Define multisampling state 
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;


		// Define blend state 
		// We define one blend state per color attachment - this example uses a 
			// single color attachment, so we only need one. Right now, we don�t do any 
			// blending, so we can ignore most of the members. 
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
			| VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		//dynamic state(...)
		VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

		VkPipelineDynamicStateCreateInfo dynamicStateInfo = {};
		dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicStateInfo.dynamicStateCount = 2;
		dynamicStateInfo.pDynamicStates = dynamicStates;

		// Create pipeline 
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2; // vertex + fragment stages 
		pipeInfo.pStages = stages;

		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr; // no tessellation 
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = nullptr; // no depth or stencil buffers 
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = &dynamicStateInfo;

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0; // first subpass of aRenderPass 

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1,
			&pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create graphics pipeline\n"
				"vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str()
			);

		}

		return lut::Pipeline(aWindow.device, pipe);

	}

	void create_swapchain_framebuffers(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers, VkImageView aDepthView)
	{
		assert(aFramebuffers.empty());


		for (std::size_t i = 0; i < aWindow.swapViews.size(); ++i)
		{
			VkImageView attachments[1] = {
				aWindow.swapViews[i]
			};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0; // normal framebuffer 
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 1;
			fbInfo.pAttachments = attachments;
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1;
			VkFramebuffer fb = VK_NULL_HANDLE;
			if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb
			); VK_SUCCESS != res)
			{

				throw lut::Error("Unable to create framebuffer for swap chain image %zu\n"
					"vkCreateFramebuffer() returned %s", i, lut::to_string(res).c_str()
				);

			}

			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));

		}


		return;


		assert(aWindow.swapViews.size() == aFramebuffers.size());
	}

	lut::DescriptorSetLayout create_scene_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0; // number must match the index of the corresponding 
		// binding = N declaration in the shader(s)!
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create descriptor set layout\n"
				"vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str()
			);

		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_object_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[2]{};
		bindings[0].binding = 0; // this must match the shaders 
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		bindings[1].binding = 1; // this must match the shaders 
		bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[1].descriptorCount = 1;
		bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = std::size(bindings);
		layoutInfo.pBindings = bindings;
		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create descriptor set layout\n"
				"vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str()
			);

		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	void create_render_descriptorSet(
		lut::VulkanWindow const& aWindow,
		VkDescriptorSet const& renderDescriptorSet,
		VkSampler const& sampler,
		VkImageView const& pbrImageView,
		VkImageView const& filterImageView)
	{
		{
			VkWriteDescriptorSet desc[2]{};

			VkDescriptorImageInfo textureInfo[2]{};
			textureInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			textureInfo[0].imageView = pbrImageView;
			textureInfo[0].sampler = sampler;

			textureInfo[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			textureInfo[1].imageView = filterImageView;
			textureInfo[1].sampler = sampler;


			desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[0].dstSet = renderDescriptorSet;
			desc[0].dstBinding = 0;
			desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[0].descriptorCount = 1;
			desc[0].pImageInfo = &textureInfo[0];

			desc[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[1].dstSet = renderDescriptorSet;
			desc[1].dstBinding = 1;
			desc[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[1].descriptorCount = 1;
			desc[1].pImageInfo = &textureInfo[1];


			constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
			vkUpdateDescriptorSets(aWindow.device, numSets, desc, 0, nullptr);
		}
	}

	void record_commands(
		VkCommandBuffer aCmdBuff,
		VkRenderPass aRenderPass,
		VkFramebuffer aFramebuffer,
		VkPipeline aGraphicsPipe,
		VkExtent2D const& aImageExtent,
		VkBuffer aSceneUBO,
		glsl::SceneUniform& aSceneUniform,
		VkPipelineLayout aGraphicsLayout,
		VkDescriptorSet aSceneDescriptors,
		VkDescriptorSet& myDescriptor,
		std::vector<TextureFragment>& bakedfragments,

		VkRenderPass compute_renderpass,
		VkFramebuffer compute_framebuffer,
		VkPipeline computePipe,
		VkPipelineLayout compute_pipeline_layout,

		VkRenderPass pbr_renderpass,
		VkFramebuffer pbr_framebuffer,
		VkPipeline pbrPipe,
		VkPipelineLayout pbr_pipeline_layout,
		VkDescriptorSet& pbr_descriptor)
	{
		VkCommandBufferBeginInfo begInfo{};
		begInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		begInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(aCmdBuff, &begInfo); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to begin recording command buffer\n""vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		// Upload scene uniforms 
		lut::buffer_barrier(aCmdBuff,
			aSceneUBO,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdUpdateBuffer(aCmdBuff, aSceneUBO, 0, sizeof(glsl::SceneUniform), &aSceneUniform);

		lut::buffer_barrier(aCmdBuff,
			aSceneUBO,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
		);


		// compute pass

		VkClearValue compute_clearValues[6]{};
		compute_clearValues[0].color.float32[0] = 0.1f; 
		compute_clearValues[0].color.float32[1] = 0.1f;
		compute_clearValues[0].color.float32[2] = 0.1f;
		compute_clearValues[0].color.float32[3] = 1.0f;

		compute_clearValues[1].color.float32[0] = 0.0f; 
		compute_clearValues[1].color.float32[1] = 0.0f;
		compute_clearValues[1].color.float32[2] = 0.0f;
		compute_clearValues[1].color.float32[3] = 1.0f;

		compute_clearValues[2].color.float32[0] = 0.0f;
		compute_clearValues[2].color.float32[1] = 0.0f;
		compute_clearValues[2].color.float32[2] = 0.0f;
		compute_clearValues[2].color.float32[3] = 1.0f;

		compute_clearValues[3].color.float32[0] = 0.0f;
		compute_clearValues[3].color.float32[1] = 0.0f;
		compute_clearValues[3].color.float32[2] = 0.0f;
		compute_clearValues[3].color.float32[3] = 1.0f;

		compute_clearValues[4].color.float32[0] = 0.1f;
		compute_clearValues[4].color.float32[1] = 0.1f;
		compute_clearValues[4].color.float32[2] = 0.1f;
		compute_clearValues[4].color.float32[3] = 1.0f;

		// depth view
		compute_clearValues[5].depthStencil.depth = 1.0f;

		VkViewport compute_viewport = {};
		compute_viewport.x = 0.0f;
		compute_viewport.y = 0.0f;
		compute_viewport.width = (float)aImageExtent.width;
		compute_viewport.height = (float)aImageExtent.height;
		compute_viewport.minDepth = 0.0f;
		compute_viewport.maxDepth = 1.0f;

		VkRect2D compute_scissor = {};
		compute_scissor.offset = { 0, 0 };
		compute_scissor.extent = aImageExtent;

		VkRenderPassBeginInfo compute_passInfo{};
		compute_passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		compute_passInfo.renderPass = compute_renderpass;
		compute_passInfo.framebuffer = compute_framebuffer;
		compute_passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
		compute_passInfo.renderArea.extent = aImageExtent;
		compute_passInfo.clearValueCount = 6;
		compute_passInfo.pClearValues = compute_clearValues;

		vkCmdBeginRenderPass(aCmdBuff, &compute_passInfo, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, computePipe);
		vkCmdSetViewport(aCmdBuff, 0, 1, &compute_viewport);
		vkCmdSetScissor(aCmdBuff, 0, 1, &compute_scissor);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, compute_pipeline_layout,0, 1, &aSceneDescriptors, 0, nullptr);
		for (size_t i = 0; i < bakedfragments.size(); i++)
		{

			PushConstantData pushConstantData;
			
			pushConstantData.baseColor_r = bakedfragments[i].baseColor.r;
			pushConstantData.baseColor_g = bakedfragments[i].baseColor.g;
			pushConstantData.baseColor_b = bakedfragments[i].baseColor.b;

			pushConstantData.emissiveColor_r = bakedfragments[i].emissiveColor.r;
			pushConstantData.emissiveColor_g = bakedfragments[i].emissiveColor.g;
			pushConstantData.emissiveColor_b = bakedfragments[i].emissiveColor.b;

			pushConstantData.roughness = bakedfragments[i].roughness;
			pushConstantData.metalness = bakedfragments[i].metalness;
			
			vkCmdPushConstants(aCmdBuff, compute_pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantData), &pushConstantData);
			
			VkBuffer indexBuffer = bakedfragments[i].indices.buffer;
			VkDeviceSize idOffset = 0;
			VkIndexType indexType = VK_INDEX_TYPE_UINT32;
			vkCmdBindIndexBuffer(aCmdBuff, indexBuffer, idOffset, indexType);

			VkBuffer buffers[3] = { bakedfragments[i].positions.buffer, bakedfragments[i].texcoords.buffer, bakedfragments[i].normals.buffer };
			VkDeviceSize offsets[3]{  };
			vkCmdBindVertexBuffers(aCmdBuff, 0, 3, buffers, offsets);

			vkCmdDrawIndexed(aCmdBuff, bakedfragments[i].idCount, 1, 0, 0, 0);
		}

		vkCmdEndRenderPass(aCmdBuff);		




		// pbr pass
		VkClearValue pbr_clearValues[2]{};
		pbr_clearValues[0].color.float32[0] = 0.1f;
		pbr_clearValues[0].color.float32[1] = 0.1f;
		pbr_clearValues[0].color.float32[2] = 0.1f;
		pbr_clearValues[0].color.float32[3] = 1.f;

		pbr_clearValues[1].color.float32[0] = 0.1f;
		pbr_clearValues[1].color.float32[1] = 0.1f;
		pbr_clearValues[1].color.float32[2] = 0.1f;
		pbr_clearValues[1].color.float32[3] = 1.f;

		VkViewport pbr_viewport = {};
		pbr_viewport.x = 0.0f;
		pbr_viewport.y = 0.0f;
		pbr_viewport.width = (float)aImageExtent.width;
		pbr_viewport.height = (float)aImageExtent.height;
		pbr_viewport.minDepth = 0.0f;
		pbr_viewport.maxDepth = 1.0f;

		VkRect2D pbr_scissor = {};
		pbr_scissor.offset = { 0, 0 };
		pbr_scissor.extent = aImageExtent;

		VkRenderPassBeginInfo pbr_passInfo{};
		pbr_passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		pbr_passInfo.renderPass = pbr_renderpass;
		pbr_passInfo.framebuffer = pbr_framebuffer;
		pbr_passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
		pbr_passInfo.renderArea.extent = VkExtent2D{ aImageExtent.width , aImageExtent.height };
		pbr_passInfo.clearValueCount = 2;
		pbr_passInfo.pClearValues = pbr_clearValues;
		vkCmdBeginRenderPass(aCmdBuff, &pbr_passInfo, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, pbrPipe);
		vkCmdSetViewport(aCmdBuff, 0, 1, &pbr_viewport);
		vkCmdSetScissor(aCmdBuff, 0, 1, &pbr_scissor);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, pbr_pipeline_layout, 0, 1, &aSceneDescriptors, 0, nullptr);

		PushBoundaryData pbr_pushBoundaryData;
		pbr_pushBoundaryData.width = aImageExtent.width;
		pbr_pushBoundaryData.height = aImageExtent.height;
		vkCmdPushConstants(aCmdBuff, pbr_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushBoundaryData), &pbr_pushBoundaryData);

		VkDescriptorSet pbr_descriptorSets[] = { pbr_descriptor };
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, pbr_pipeline_layout, 1, 1, pbr_descriptorSets, 0, nullptr);
		vkCmdDraw(aCmdBuff, 3, 1, 0, 0);
		vkCmdEndRenderPass(aCmdBuff);





		// render pass
		VkClearValue clearValues[1]{};
		clearValues[0].color.float32[0] = 0.1f;
		clearValues[0].color.float32[1] = 0.1f;
		clearValues[0].color.float32[2] = 0.1f;
		clearValues[0].color.float32[3] = 1.f;

		VkViewport viewport = {};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)aImageExtent.width;
		viewport.height = (float)aImageExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = aImageExtent;

		VkRenderPassBeginInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo.renderPass = aRenderPass;
		passInfo.framebuffer = aFramebuffer;
		passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
		passInfo.renderArea.extent = VkExtent2D{ aImageExtent.width , aImageExtent.height };
		passInfo.clearValueCount = 1;
		passInfo.pClearValues = clearValues;
		vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsPipe);
		vkCmdSetViewport(aCmdBuff, 0, 1, &viewport);
		vkCmdSetScissor(aCmdBuff, 0, 1, &scissor);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 0, 1, &aSceneDescriptors, 0, nullptr);

		PushBoundaryData pushBoundaryData;
		pushBoundaryData.width = aImageExtent.width;
		pushBoundaryData.height = aImageExtent.height;
		vkCmdPushConstants(aCmdBuff, aGraphicsLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushBoundaryData), &pushBoundaryData);		

		VkDescriptorSet descriptorSets[] = { myDescriptor };
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 1, 1, descriptorSets, 0, nullptr);			
		vkCmdDraw(aCmdBuff, 3, 1, 0, 0);
		vkCmdEndRenderPass(aCmdBuff);




		VkImageSubresourceLayers layers{};
		layers.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		layers.mipLevel = 0;
		layers.baseArrayLayer = 0;
		layers.layerCount = 1;

		if (auto const res = vkEndCommandBuffer(aCmdBuff); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to end recording command buffer\n""vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}



	}

	void submit_commands(lut::VulkanWindow const& aWindow, VkCommandBuffer aCmdBuff, VkFence aFence, VkSemaphore aWaitSemaphore, VkSemaphore aSignalSemaphore)
	{
		VkPipelineStageFlags waitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &aCmdBuff;

		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &aWaitSemaphore;
		submitInfo.pWaitDstStageMask = &waitPipelineStages;
		
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &aSignalSemaphore;

		if (auto const res = vkQueueSubmit(aWindow.graphicsQueue, 1, &submitInfo, aFence); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to submit command buffer to queue\n" "vkQueueSubmit() returned %s", lut::to_string(res).c_str());		
		}		
		return;
	}

	void present_results(VkQueue aPresentQueue, VkSwapchainKHR aSwapchain, std::uint32_t aImageIndex, VkSemaphore aRenderFinished, bool& aNeedToRecreateSwapchain)
	{
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &aRenderFinished;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &aSwapchain;
		presentInfo.pImageIndices = &aImageIndex;
		presentInfo.pResults = nullptr;

		auto const presentRes = vkQueuePresentKHR(aPresentQueue, &presentInfo);

		if (VK_SUBOPTIMAL_KHR == presentRes || VK_ERROR_OUT_OF_DATE_KHR == presentRes)
		{
			aNeedToRecreateSwapchain = true;
		}
		else if (VK_SUCCESS != presentRes)
		{
			throw lut::Error("Unable present swapchain image %u\n" "vkQueuePresentKHR() returned %s", aImageIndex, lut::to_string(presentRes).c_str());
		}
	}

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator)
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = cfg::kDepthFormat;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (auto const res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to allocate depth buffer image.\n" "vmaCreateImage() returned %s", lut::to_string(res).c_str());
		}

		lut::Image depthImage(aAllocator.allocator, image, allocation);

		// Create the image view 
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = depthImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = cfg::kDepthFormat;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
		 VK_IMAGE_ASPECT_DEPTH_BIT,
		 0, 1,
		 0, 1
		};

		VkImageView view = VK_NULL_HANDLE;
		if (auto const res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create image view\n" "vkCreateImageView() returned %s", lut::to_string(res).c_str());
		}

		return{ std::move(depthImage), lut::ImageView(aWindow.device, view) };
	}
}


namespace 
{
	lut::RenderPass create_compute_pass(lut::VulkanWindow const& aWindow)
	{
		VkAttachmentDescription attachments[6]{};
		attachments[0].format = VK_FORMAT_B8G8R8A8_SRGB; //changed! 
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; //changed! 

		attachments[1].format = VK_FORMAT_R32G32B32A32_SFLOAT; //changed! 
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; //changed! 

		attachments[2].format = VK_FORMAT_R32G32B32A32_SFLOAT; //changed! 
		attachments[2].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[2].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; //changed! 

		attachments[3].format = VK_FORMAT_B8G8R8A8_SRGB; //changed! 
		attachments[3].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[3].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[3].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[3].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[3].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; //changed! 

		attachments[4].format = VK_FORMAT_B8G8R8A8_SRGB; //changed! 
		attachments[4].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[4].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[4].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[4].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[4].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; //changed! 

		attachments[5].format = cfg::kDepthFormat;
		attachments[5].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[5].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[5].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[5].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[5].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference subpassAttachments[5]{};
		subpassAttachments[0].attachment = 0; // this refers to attachments[0] 
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		subpassAttachments[1].attachment = 1; // this refers to attachments[0] 
		subpassAttachments[1].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		subpassAttachments[2].attachment = 2; // this refers to attachments[0] 
		subpassAttachments[2].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		subpassAttachments[3].attachment = 3; // this refers to attachments[0] 
		subpassAttachments[3].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		subpassAttachments[4].attachment = 4; // this refers to attachments[0] 
		subpassAttachments[4].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		// New: 
		VkAttachmentReference depthAttachment{};
		depthAttachment.attachment = 5; // this refers to attachments[1] 
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 5;
		subpasses[0].pColorAttachments = subpassAttachments;
		subpasses[0].pDepthStencilAttachment = &depthAttachment; // New!



		VkSubpassDependency dep1 = {};
		dep1.srcSubpass = VK_SUBPASS_EXTERNAL;
		dep1.dstSubpass = 0;
		dep1.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dep1.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dep1.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dep1.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dep1.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkSubpassDependency dep2 = {};
		dep2.srcSubpass = 0;
		dep2.dstSubpass = VK_SUBPASS_EXTERNAL;
		dep2.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dep2.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dep2.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dep2.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dep2.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkSubpassDependency dependencies[2] = { dep1, dep2 };


		// changed: no explicit subpass dependencies 

		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 6;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 2; //changed! 
		passInfo.pDependencies = dependencies; //changed! 

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create render pass\n" "vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);
	}

	lut::PipelineLayout create_compute_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout const& aSceneLayout, VkDescriptorSetLayout const aObjectLayout)
	{

		VkDescriptorSetLayout layouts[] = {
			// Order must match the set = N in the shaders 
			aSceneLayout,
			aObjectLayout
		};

		VkPushConstantRange pushConstantRange{};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; // Or the stage where the float value will be used
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(PushConstantData);

		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]); // updated!
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 1;
		layoutInfo.pPushConstantRanges = &pushConstantRange;


		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr,
			&layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n" "vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
		}
		return lut::PipelineLayout(aContext.device, layout);
	}

	lut::Pipeline create_compute_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{

		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::v_ComputeShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::f_ComputeShaderPath);

		


		// Define shader stages in the pipeline 
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";


		//====================================================================================================================================
		VkVertexInputBindingDescription vertexInputs[4]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 2;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[2].binding = 2;
		vertexInputs[2].stride = sizeof(float) * 3;
		vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[3]{};
		vertexAttributes[0].binding = 0; // must match binding above 
		vertexAttributes[0].location = 0; // must match shader 
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		vertexAttributes[1].binding = 1; // must match binding above 
		vertexAttributes[1].location = 1; // must match shader 
		vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[1].offset = 0;

		vertexAttributes[2].binding = 2; // must match binding above 
		vertexAttributes[2].location = 2; // must match shader 
		vertexAttributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[2].offset = 0;


		//====================================================================================================================================


		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 3; // number of vertexInputs above 
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 3; // number of vertexAttributes above 
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		// Define viewport and scissor regions 
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0, 0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width, aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// Define rasterization options 
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.f; // required.

		// Define multisampling state 
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;


		// Define blend state 
		// We define one blend state per color attachment - this example uses a 
			// single color attachment, so we only need one. Right now, we don�t do any 
			// blending, so we can ignore most of the members. 
		VkPipelineColorBlendAttachmentState blendStates[5]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
			| VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		blendStates[1].blendEnable = VK_FALSE;
		blendStates[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
			| VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		blendStates[2].blendEnable = VK_FALSE;
		blendStates[2].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
			| VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		blendStates[3].blendEnable = VK_FALSE;
		blendStates[3].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
			| VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		blendStates[4].blendEnable = VK_FALSE;
		blendStates[4].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
			| VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 5;
		blendInfo.pAttachments = blendStates;

		//dynamic state(...)
		VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

		VkPipelineDynamicStateCreateInfo dynamicStateInfo = {};
		dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicStateInfo.dynamicStateCount = 2;
		dynamicStateInfo.pDynamicStates = dynamicStates;

		// Create pipeline 
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2; // vertex + fragment stages 
		pipeInfo.pStages = stages;

		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr; // no tessellation 
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo; // no depth or stencil buffers 
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = &dynamicStateInfo;

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0; // first subpass of aRenderPass 

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1,
			&pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create graphics pipeline\n"
				"vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str()
			);

		}

		return lut::Pipeline(aWindow.device, pipe);

	}

	std::tuple<lut::Image, lut::ImageView> create_empty_image_view(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator, VkFormat format)
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = format;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (auto const res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo, &image,
			&allocation, nullptr); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create framebuffer for offline renderering"
				"vkCreateImage reutrned %s", lut::to_string(res));
		}

		lut::Image baseImage(aAllocator.allocator, image, allocation);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = baseImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = format;
		viewInfo.components = VkComponentMapping{
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY
		};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_COLOR_BIT,
			0, 1,
			0, 1
		};

		VkImageView imageView = VK_NULL_HANDLE;

		if (auto const res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &imageView);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create image view for offline framebuffer" "vkCreateImageView returned %s\n", lut::to_string(res));
		}

		return { std::move(baseImage), lut::ImageView(aWindow.device, imageView) };
	}

	void create_compute_framebuffer(
		lut::VulkanWindow const& aWindow, 
		VkRenderPass aRenderPass, 
		lut::Framebuffer& aFramebuffers, 
		VkImageView aDepthView, 
		VkImageView emptyView,
		VkImageView positionView,
		VkImageView normalView,
		VkImageView materialView,
		VkImageView emissiveView

		)
	{
		VkImageView attachments[6] = {
			emptyView,
			positionView,
			normalView,
			materialView,
			emissiveView,
			aDepthView
		};

		VkFramebufferCreateInfo fbInfo{};
		fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fbInfo.flags = 0;
		fbInfo.renderPass = aRenderPass;
		fbInfo.attachmentCount = 6;
		fbInfo.pAttachments = attachments;
		fbInfo.width = aWindow.swapchainExtent.width;
		fbInfo.height = aWindow.swapchainExtent.height;
		fbInfo.layers = 1;

		VkFramebuffer fb = VK_NULL_HANDLE;
		if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo,
			nullptr, &fb); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create framebuffer for swap chain\n"
				"vkCreateFramebuffer() returned %s",
				lut::to_string(res).c_str());
		}

		aFramebuffers = lut::Framebuffer(aWindow.device, fb);
	}

	lut::DescriptorSetLayout create_compute_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = 0;
		layoutInfo.pBindings = nullptr;
		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create descriptor set layout\n"
				"vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str()
			);

		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	

	
}

namespace
{
	lut::RenderPass create_pbr_pass(lut::VulkanWindow const& aWindow)
	{
		VkAttachmentDescription attachments[2]{};
		attachments[0].format = VK_FORMAT_B8G8R8A8_SRGB; //changed! 
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; //changed! 

		attachments[1].format = VK_FORMAT_B8G8R8A8_SRGB; //changed! 
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; //changed! 

		VkAttachmentReference subpassAttachments[2]{};
		subpassAttachments[0].attachment = 0; // this refers to attachments[0] 
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		subpassAttachments[1].attachment = 1; // this refers to attachments[0] 
		subpassAttachments[1].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 2;
		subpasses[0].pColorAttachments = subpassAttachments;
		subpasses[0].pDepthStencilAttachment = nullptr; // New!



		VkSubpassDependency dep1 = {};
		dep1.srcSubpass = VK_SUBPASS_EXTERNAL;
		dep1.dstSubpass = 0;
		dep1.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dep1.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dep1.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dep1.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dep1.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkSubpassDependency dep2 = {};
		dep2.srcSubpass = 0;
		dep2.dstSubpass = VK_SUBPASS_EXTERNAL;
		dep2.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dep2.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dep2.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dep2.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dep2.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkSubpassDependency dependencies[2] = { dep1, dep2 };


		// changed: no explicit subpass dependencies 

		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 2;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 2; //changed! 
		passInfo.pDependencies = dependencies; //changed! 

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create render pass\n" "vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);
	}

	lut::DescriptorSetLayout create_pbr_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[5]{};
		bindings[0].binding = 0; // this must match the shaders 
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		bindings[1].binding = 1; // this must match the shaders 
		bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[1].descriptorCount = 1;
		bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		bindings[2].binding = 2; // this must match the shaders 
		bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[2].descriptorCount = 1;
		bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		bindings[3].binding = 3; // this must match the shaders 
		bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[3].descriptorCount = 1;
		bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		bindings[4].binding = 4; // this must match the shaders 
		bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[4].descriptorCount = 1;
		bindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = std::size(bindings);
		layoutInfo.pBindings = bindings;
		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create descriptor set layout\n"
				"vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str()
			);

		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::PipelineLayout create_pbr_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout const& aSceneLayout, VkDescriptorSetLayout const& aObjectLayout)
	{

		VkDescriptorSetLayout layouts[] = {
			// Order must match the set = N in the shaders 
			aSceneLayout,
			aObjectLayout
		};

		VkPushConstantRange pushConstantRange{};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT; // Or the stage where the float value will be used
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(PushBoundaryData);

		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]); // updated!
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 1;
		layoutInfo.pPushConstantRanges = &pushConstantRange;


		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr,
			&layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n" "vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
		}
		return lut::PipelineLayout(aContext.device, layout);
	}

	lut::Pipeline create_pbr_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::v_pbrShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::f_pbrShaderPath);

		// Define shader stages in the pipeline 
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		//====================================================================================================================================


		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;



		// Define viewport and scissor regions 
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0, 0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width, aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// Define rasterization options 
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.f; // required.

		// Define multisampling state 
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;


		// Define blend state 
		// We define one blend state per color attachment - this example uses a 
			// single color attachment, so we only need one. Right now, we don�t do any 
			// blending, so we can ignore most of the members. 
		VkPipelineColorBlendAttachmentState blendStates[2]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
			| VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		blendStates[1].blendEnable = VK_FALSE;
		blendStates[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
			| VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 2;
		blendInfo.pAttachments = blendStates;

		//dynamic state(...)
		VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

		VkPipelineDynamicStateCreateInfo dynamicStateInfo = {};
		dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicStateInfo.dynamicStateCount = 2;
		dynamicStateInfo.pDynamicStates = dynamicStates;

		// Create pipeline 
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2; // vertex + fragment stages 
		pipeInfo.pStages = stages;

		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr; // no tessellation 
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = nullptr; // no depth or stencil buffers 
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = &dynamicStateInfo;

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0; // first subpass of aRenderPass 

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1,
			&pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create graphics pipeline\n"
				"vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str()
			);

		}

		return lut::Pipeline(aWindow.device, pipe);

	}

	void create_pbr_framebuffer(
		lut::VulkanWindow const& aWindow,
		VkRenderPass aRenderPass,
		lut::Framebuffer& aFramebuffers,
		VkImageView pbrView,
		VkImageView filterView
	)
	{
		VkImageView attachments[2] = {
			pbrView,
			filterView
		};

		VkFramebufferCreateInfo fbInfo{};
		fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fbInfo.flags = 0;
		fbInfo.renderPass = aRenderPass;
		fbInfo.attachmentCount = 2;
		fbInfo.pAttachments = attachments;
		fbInfo.width = aWindow.swapchainExtent.width;
		fbInfo.height = aWindow.swapchainExtent.height;
		fbInfo.layers = 1;

		VkFramebuffer fb = VK_NULL_HANDLE;
		if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo,
			nullptr, &fb); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create framebuffer for swap chain\n"
				"vkCreateFramebuffer() returned %s",
				lut::to_string(res).c_str());
		}

		aFramebuffers = lut::Framebuffer(aWindow.device, fb);
	}

	void create_pbr_descriptorSet(
		lut::VulkanWindow const& aWindow,
		VkDescriptorSet const& renderDescriptorSet,
		VkImageView const& baseImageView,
		VkSampler const& sampler,
		VkImageView const& positionImageView,
		VkImageView const& normalImageView,
		VkImageView const& materialImageView,
		VkImageView const& emissiveImageView)
	{
		{
			VkWriteDescriptorSet desc[5]{};

			VkDescriptorImageInfo textureInfo[5]{};
			textureInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			textureInfo[0].imageView = baseImageView;
			textureInfo[0].sampler = sampler;

			textureInfo[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			textureInfo[1].imageView = positionImageView;
			textureInfo[1].sampler = sampler;

			textureInfo[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			textureInfo[2].imageView = normalImageView;
			textureInfo[2].sampler = sampler;

			textureInfo[3].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			textureInfo[3].imageView = materialImageView;
			textureInfo[3].sampler = sampler;

			textureInfo[4].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			textureInfo[4].imageView = emissiveImageView;
			textureInfo[4].sampler = sampler;


			desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[0].dstSet = renderDescriptorSet;
			desc[0].dstBinding = 0;
			desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[0].descriptorCount = 1;
			desc[0].pImageInfo = &textureInfo[0];

			desc[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[1].dstSet = renderDescriptorSet;
			desc[1].dstBinding = 1;
			desc[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[1].descriptorCount = 1;
			desc[1].pImageInfo = &textureInfo[1];

			desc[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[2].dstSet = renderDescriptorSet;
			desc[2].dstBinding = 2;
			desc[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[2].descriptorCount = 1;
			desc[2].pImageInfo = &textureInfo[2];

			desc[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[3].dstSet = renderDescriptorSet;
			desc[3].dstBinding = 3;
			desc[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[3].descriptorCount = 1;
			desc[3].pImageInfo = &textureInfo[3];

			desc[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[4].dstSet = renderDescriptorSet;
			desc[4].dstBinding = 4;
			desc[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[4].descriptorCount = 1;
			desc[4].pImageInfo = &textureInfo[4];


			constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
			vkUpdateDescriptorSets(aWindow.device, numSets, desc, 0, nullptr);
		}
	}
}
//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 