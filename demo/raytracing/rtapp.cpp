#include "rtapp.h"

#include "glm/gtc/matrix_transform.hpp"

RayTracingApp::RayTracingApp(uint32_t width, uint32_t height) {
  grassland::vulkan::framework::CoreSettings core_settings;
  core_settings.window_width = width;
  core_settings.window_height = height;
  core_settings.raytracing_pipeline_required = true;
  core_settings.validation_layer = true;
  core_ = std::make_unique<grassland::vulkan::framework::Core>(core_settings);
}

void RayTracingApp::Run() {
  OnInit();
  while (!glfwWindowShouldClose(core_->GetWindow())) {
    OnLoop();
    glfwPollEvents();
  }
  OnClose();
}

void RayTracingApp::OnInit() {
  frame_texture_ = std::make_unique<grassland::vulkan::framework::TextureImage>(
      core_.get(), core_->GetFramebufferWidth(), core_->GetFramebufferHeight(),
      VK_FORMAT_B8G8R8A8_UNORM);
  core_->SetFrameSizeCallback(
      [this](int width, int height) { frame_texture_->Resize(width, height); });

  std::vector<glm::vec3> vertices = {
      {1.0f, 1.0f, 0.0f}, {-1.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f}};
  std::vector<uint32_t> indices = {0, 1, 2};
  bottom_level_acceleration_structure_ = std::make_unique<
      grassland::vulkan::raytracing::BottomLevelAccelerationStructure>(
      core_->GetDevice(), core_->GetCommandPool(), vertices.data(),
      sizeof(glm::vec3) * vertices.size(), indices.data(), indices.size(),
      sizeof(glm::vec3));
  top_level_acceleration_structure_ = std::make_unique<
      grassland::vulkan::raytracing::TopLevelAccelerationStructure>(
      core_->GetDevice(), core_->GetCommandPool(),
      std::vector<std::pair<
          grassland::vulkan::raytracing::BottomLevelAccelerationStructure *,
          glm::mat4>>{
          {bottom_level_acceleration_structure_.get(), glm::mat4{1.0f}}});
  camera_object_buffer_ = std::make_unique<
      grassland::vulkan::framework::StaticBuffer<CameraObject>>(core_.get(), 1);
  CameraObject camera_object{};
  camera_object.screen_to_camera =
      glm::inverse(glm::perspectiveLH(glm::radians(90.0f),
                                      (float)core_->GetFramebufferWidth() /
                                          (float)core_->GetFramebufferHeight(),
                                      0.1f, 10.0f));
  camera_object.camera_to_world = glm::inverse(
      glm::lookAtLH(glm::vec3{0.0f, 0.0f, -2.0f}, glm::vec3{0.0f, 0.0f, 0.0f},
                    glm::vec3{0.0f, 1.0f, 0.0f}));
  camera_object_buffer_->Upload(&camera_object);
  grassland::vulkan::helper::DescriptorSetLayoutBindings
      descriptorSetLayoutBindings;
  descriptorSetLayoutBindings.AddBinding(
      0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
      VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr);
  descriptorSetLayoutBindings.AddBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                         VK_SHADER_STAGE_RAYGEN_BIT_KHR);
  descriptorSetLayoutBindings.AddBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                         1, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
  descriptor_set_layout_ =
      std::make_unique<grassland::vulkan::DescriptorSetLayout>(
          core_->GetDevice(), descriptorSetLayoutBindings);
  pipeline_layout_ = std::make_unique<grassland::vulkan::PipelineLayout>(
      core_->GetDevice(), descriptor_set_layout_.get());
  descriptor_pool_ = std::make_unique<grassland::vulkan::DescriptorPool>(
      core_->GetDevice(), descriptorSetLayoutBindings, 1);
  descriptor_set_ = std::make_unique<grassland::vulkan::DescriptorSet>(
      descriptor_set_layout_.get(), descriptor_pool_.get());

  VkWriteDescriptorSetAccelerationStructureKHR
      descriptor_acceleration_structure_info{};
  descriptor_acceleration_structure_info.sType =
      VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
  descriptor_acceleration_structure_info.accelerationStructureCount = 1;
  descriptor_acceleration_structure_info.pAccelerationStructures =
      &top_level_acceleration_structure_->GetHandle();

  VkWriteDescriptorSet acceleration_structure_write{};
  acceleration_structure_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  acceleration_structure_write.dstSet = descriptor_set_->GetHandle();
  acceleration_structure_write.dstBinding = 0;
  acceleration_structure_write.descriptorCount = 1;
  acceleration_structure_write.descriptorType =
      VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  // The acceleration structure descriptor has to be chained via pNext
  acceleration_structure_write.pNext = &descriptor_acceleration_structure_info;

  VkDescriptorBufferInfo buffer_info{};
  buffer_info.buffer = camera_object_buffer_->GetBuffer(0)->GetHandle();
  buffer_info.offset = 0;
  buffer_info.range = camera_object_buffer_->BufferSize();

  VkDescriptorImageInfo image_info{};
  image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  image_info.imageView = frame_texture_->GetImageView()->GetHandle();
  image_info.sampler = nullptr;

  VkWriteDescriptorSet result_image_write{};
  result_image_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  result_image_write.dstSet = descriptor_set_->GetHandle();
  result_image_write.dstBinding = 1;
  result_image_write.descriptorCount = 1;
  result_image_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  result_image_write.pImageInfo = &image_info;

  VkWriteDescriptorSet camera_object_write{};
  camera_object_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  camera_object_write.dstSet = descriptor_set_->GetHandle();
  camera_object_write.dstBinding = 2;
  camera_object_write.descriptorCount = 1;
  camera_object_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  camera_object_write.pBufferInfo = &buffer_info;

  std::vector<VkWriteDescriptorSet> write_descriptor_sets{
      acceleration_structure_write, result_image_write, camera_object_write};
  vkUpdateDescriptorSets(core_->GetDevice()->GetHandle(),
                         write_descriptor_sets.size(),
                         write_descriptor_sets.data(), 0, nullptr);

  grassland::vulkan::ShaderModule ray_gen_shader(
      core_->GetDevice(), "../shaders/raytracing.rgen.spv");
  grassland::vulkan::ShaderModule closest_hit_shader(
      core_->GetDevice(), "../shaders/raytracing.rchit.spv");
  grassland::vulkan::ShaderModule miss_shader(
      core_->GetDevice(), "../shaders/raytracing.rmiss.spv");

  ray_tracing_pipeline_ =
      std::make_unique<grassland::vulkan::raytracing::RayTracingPipeline>(
          core_->GetDevice(), pipeline_layout_.get(), ray_gen_shader,
          closest_hit_shader, miss_shader);
  shader_binding_table_ =
      std::make_unique<grassland::vulkan::raytracing::ShaderBindingTable>(
          ray_tracing_pipeline_.get());
}

void RayTracingApp::OnLoop() {
  OnUpdate();
  OnRender();
}

void RayTracingApp::OnClose() {
  core_->GetDevice()->WaitIdle();
  // frame_texture_.reset();
}

void RayTracingApp::OnUpdate() {
}

void RayTracingApp::OnRender() {
  core_->BeginCommandRecord();
  frame_texture_->ClearColor({0.6f, 0.7f, 0.8f, 1.0f});
  auto draw_cmd_buffers = core_->GetCommandBuffer()->GetHandle();
  vkCmdBindPipeline(draw_cmd_buffers, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                    ray_tracing_pipeline_->GetHandle());
  vkCmdBindDescriptorSets(
      draw_cmd_buffers, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
      pipeline_layout_->GetHandle(), 0, 1, &descriptor_set_->GetHandle(), 0, 0);

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR
      ray_tracing_pipeline_properties =
          core_->GetPhysicalDevice()->GetRayTracingProperties();
  auto aligned_size = [](uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
  };
  const uint32_t handle_size_aligned =
      aligned_size(ray_tracing_pipeline_properties.shaderGroupHandleSize,
                   ray_tracing_pipeline_properties.shaderGroupHandleAlignment);

  VkStridedDeviceAddressRegionKHR ray_gen_shader_sbt_entry{};
  ray_gen_shader_sbt_entry.deviceAddress =
      shader_binding_table_->GetRayGenDeviceAddress();
  ray_gen_shader_sbt_entry.stride = handle_size_aligned;
  ray_gen_shader_sbt_entry.size = handle_size_aligned;

  VkStridedDeviceAddressRegionKHR miss_shader_sbt_entry{};
  miss_shader_sbt_entry.deviceAddress =
      shader_binding_table_->GetMissDeviceAddress();
  miss_shader_sbt_entry.stride = handle_size_aligned;
  miss_shader_sbt_entry.size = handle_size_aligned;

  VkStridedDeviceAddressRegionKHR hit_shader_sbt_entry{};
  hit_shader_sbt_entry.deviceAddress =
      shader_binding_table_->GetClosestHitDeviceAddress();
  hit_shader_sbt_entry.stride = handle_size_aligned;
  hit_shader_sbt_entry.size = handle_size_aligned;

  VkStridedDeviceAddressRegionKHR callable_shader_sbt_entry{};

  vkCmdTraceRaysKHR(draw_cmd_buffers, &ray_gen_shader_sbt_entry,
                    &miss_shader_sbt_entry, &hit_shader_sbt_entry,
                    &callable_shader_sbt_entry, core_->GetFramebufferWidth(),
                    core_->GetFramebufferHeight(), 1);
  core_->Output(frame_texture_.get());
  core_->EndCommandRecordAndSubmit();
}
