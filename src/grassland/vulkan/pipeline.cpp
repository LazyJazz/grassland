#include <grassland/file/file.h>
#include <grassland/logging/logging.h>
#include <grassland/vulkan/pipeline.h>

namespace grassland::vulkan {
Pipeline::Pipeline(Device *device,
                   RenderPass *render_pass,
                   PipelineLayout *pipeline_layout,
                   ShaderModule *vertex_shader,
                   ShaderModule *fragment_shader,
                   ShaderModule *geometry_shader)
    : handle_{} {
  ConstructorCommon(device, render_pass, pipeline_layout, vertex_shader,
                    fragment_shader, geometry_shader);
}

Pipeline::Pipeline(Device *device,
                   RenderPass *render_pass,
                   PipelineLayout *pipeline_layout,
                   const char *vertex_shader_file_path,
                   const char *fragment_shader_file_path,
                   const char *geometry_shader_file_path)
    : Pipeline(device,
               render_pass,
               pipeline_layout,
               file::ReadFileBinary(vertex_shader_file_path),
               file::ReadFileBinary(fragment_shader_file_path),
               geometry_shader_file_path
                   ? file::ReadFileBinary(geometry_shader_file_path)
                   : std::vector<uint8_t>{}) {
}

Pipeline::Pipeline(Device *device,
                   RenderPass *render_pass,
                   PipelineLayout *pipeline_layout,
                   const std::vector<uint8_t> &vertex_shader_spv_data,
                   const std::vector<uint8_t> &fragment_shader_spv_data,
                   const std::vector<uint8_t> &geometry_shader_spv_data)
    : handle_{} {
  std::unique_ptr<ShaderModule> vertex_shader =
      std::make_unique<ShaderModule>(device, vertex_shader_spv_data);
  std::unique_ptr<ShaderModule> fragment_shader =
      std::make_unique<ShaderModule>(device, fragment_shader_spv_data);
  std::unique_ptr<ShaderModule> geometry_shader;
  if (!geometry_shader_spv_data.empty()) {
    geometry_shader =
        std::make_unique<ShaderModule>(device, geometry_shader_spv_data);
  }
  ConstructorCommon(device, render_pass, pipeline_layout, vertex_shader.get(),
                    fragment_shader.get(), geometry_shader.get());
}

void Pipeline::ConstructorCommon(Device *device,
                                 RenderPass *render_pass,
                                 PipelineLayout *pipeline_layout,
                                 ShaderModule *vertex_shader,
                                 ShaderModule *fragment_shader,
                                 ShaderModule *geometry_shader) {
  device_ = device;

  VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
  vertShaderStageInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vertShaderStageInfo.module = vertex_shader->GetHandle();
  vertShaderStageInfo.pName = "main";

  VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
  fragShaderStageInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragShaderStageInfo.module = fragment_shader->GetHandle();
  fragShaderStageInfo.pName = "main";

  std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {
      vertShaderStageInfo, fragShaderStageInfo};

  VkPipelineShaderStageCreateInfo geomShaderStageInfo{};
  if (geometry_shader) {
    geomShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    geomShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    geomShaderStageInfo.module = geometry_shader->GetHandle();
    geomShaderStageInfo.pName = "main";
    shaderStages.push_back(geomShaderStageInfo);
  }

  VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
  vertexInputInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputInfo.vertexBindingDescriptionCount = 0;
  vertexInputInfo.vertexAttributeDescriptionCount = 0;

  VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  VkPipelineViewportStateCreateInfo viewportState{};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendAttachmentState colorBlendAttachment{};
  colorBlendAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo colorBlending{};
  colorBlending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = VK_LOGIC_OP_COPY;
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;
  colorBlending.blendConstants[0] = 0.0f;
  colorBlending.blendConstants[1] = 0.0f;
  colorBlending.blendConstants[2] = 0.0f;
  colorBlending.blendConstants[3] = 0.0f;

  std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                               VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamicState{};
  dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
  dynamicState.pDynamicStates = dynamicStates.data();

  VkGraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
  pipelineInfo.pStages = shaderStages.data();
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDynamicState = &dynamicState;
  pipelineInfo.layout = pipeline_layout->GetHandle();
  pipelineInfo.renderPass = render_pass->GetHandle();
  pipelineInfo.subpass = 0;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

  if (vkCreateGraphicsPipelines(device_->GetHandle(), VK_NULL_HANDLE, 1,
                                &pipelineInfo, nullptr,
                                &handle_) != VK_SUCCESS) {
    LAND_ERROR("Vulkan failed to create graphics pipeline!");
  }
}

Pipeline::~Pipeline() {
  vkDestroyPipeline(device_->GetHandle(), handle_, nullptr);
}
}  // namespace grassland::vulkan
