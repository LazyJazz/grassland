#include "grassland/vulkan/pipeline/pipeline.h"

#include <utility>

namespace grassland::vulkan {
Pipeline::Pipeline(struct Core *core, PipelineSettings settings)
    : core_(core), settings_(std::move(settings)) {
  VkPipelineVertexInputStateCreateInfo vertex_input_info{};
  vertex_input_info.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertex_input_info.vertexBindingDescriptionCount =
      settings_.vertex_input_binding_descriptions.size();
  if (vertex_input_info.vertexBindingDescriptionCount) {
    vertex_input_info.pVertexBindingDescriptions =
        settings_.vertex_input_binding_descriptions.data();
  }
  vertex_input_info.vertexAttributeDescriptionCount =
      settings_.vertex_input_attribute_descriptions.size();
  if (vertex_input_info.vertexAttributeDescriptionCount) {
    vertex_input_info.pVertexAttributeDescriptions =
        settings_.vertex_input_attribute_descriptions.data();
  }

  VkPipelineViewportStateCreateInfo viewport_state{};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.pViewports = nullptr;  // Dynamic
  viewport_state.scissorCount = 1;
  viewport_state.pScissors = nullptr;  // Dynamic

  VkPipelineColorBlendStateCreateInfo color_blend_state{};
  color_blend_state.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blend_state.logicOpEnable = VK_FALSE;
  color_blend_state.logicOp = VK_LOGIC_OP_COPY;
  color_blend_state.attachmentCount =
      settings_.pipeline_color_blend_attachment_states.size();
  color_blend_state.pAttachments =
      settings_.pipeline_color_blend_attachment_states.data();
  color_blend_state.blendConstants[0] = 0.0f;
  color_blend_state.blendConstants[1] = 0.0f;
  color_blend_state.blendConstants[2] = 0.0f;
  color_blend_state.blendConstants[3] = 0.0f;

  std::vector<VkDynamicState> dynamic_states = {VK_DYNAMIC_STATE_VIEWPORT,
                                                VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic_state{};
  dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic_state.dynamicStateCount =
      static_cast<uint32_t>(dynamic_states.size());
  if (dynamic_state.dynamicStateCount) {
    dynamic_state.pDynamicStates = dynamic_states.data();
  }

  VkGraphicsPipelineCreateInfo pipeline_create_info{};
  pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_create_info.stageCount = settings_.shader_stage_create_infos.size();
  pipeline_create_info.pStages = settings_.shader_stage_create_infos.data();
  pipeline_create_info.pVertexInputState = &vertex_input_info;
  pipeline_create_info.pInputAssemblyState =
      &settings_.input_assembly_state_create_info;
  pipeline_create_info.pViewportState = &viewport_state;
  pipeline_create_info.pRasterizationState =
      &settings_.rasterization_state_create_info;
  pipeline_create_info.pMultisampleState =
      &settings_.multisample_state_create_info;
  pipeline_create_info.pColorBlendState = &color_blend_state;
  pipeline_create_info.pDynamicState = &dynamic_state;
  pipeline_create_info.layout = settings_.pipeline_layout->Handle();
  pipeline_create_info.renderPass = settings_.render_pass->Handle();
  pipeline_create_info.pDepthStencilState =
      settings_.depth_stencil_state_create_info.has_value()
          ? &settings_.depth_stencil_state_create_info.value()
          : nullptr;
  pipeline_create_info.subpass = 0;
  pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;

  if (vkCreateGraphicsPipelines(core_->Device()->Handle(), VK_NULL_HANDLE, 1,
                                &pipeline_create_info, nullptr,
                                &pipeline_) != VK_SUCCESS) {
    LAND_ERROR("[Vulkan] failed to create graphics pipeline!");
  }
}

Pipeline::~Pipeline() {
  vkDestroyPipeline(core_->Device()->Handle(), pipeline_, nullptr);
}

VkPipeline Pipeline::Handle() const {
  return pipeline_;
}

struct Core *Pipeline::Core() const {
  return core_;
}

PipelineSettings::PipelineSettings(RenderPass *render_pass,
                                   PipelineLayout *pipeline_layout)
    : render_pass(render_pass), pipeline_layout(pipeline_layout) {
  input_assembly_state_create_info.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  input_assembly_state_create_info.topology =
      VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  input_assembly_state_create_info.primitiveRestartEnable = VK_FALSE;

  multisample_state_create_info.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample_state_create_info.sampleShadingEnable = VK_FALSE;
  multisample_state_create_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  multisample_state_create_info.minSampleShading = 1.0f;
  multisample_state_create_info.pSampleMask = nullptr;
  multisample_state_create_info.alphaToCoverageEnable = VK_FALSE;
  multisample_state_create_info.alphaToOneEnable = VK_FALSE;

  rasterization_state_create_info.sType =
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterization_state_create_info.depthClampEnable = VK_FALSE;
  rasterization_state_create_info.rasterizerDiscardEnable = VK_FALSE;
  rasterization_state_create_info.polygonMode = VK_POLYGON_MODE_FILL;
  rasterization_state_create_info.lineWidth = 1.0f;
  rasterization_state_create_info.cullMode = VK_CULL_MODE_NONE;
  rasterization_state_create_info.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rasterization_state_create_info.depthBiasEnable = VK_FALSE;

  if (render_pass) {
    if (render_pass->DepthAttachmentReference().has_value()) {
      depth_stencil_state_create_info = VkPipelineDepthStencilStateCreateInfo{
          VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
          nullptr,
          0,
          VK_TRUE,
          VK_TRUE,
          VK_COMPARE_OP_LESS,
          VK_FALSE,
          VK_FALSE,
          VkStencilOpState{},
          VkStencilOpState{},
          0.0f,
          1.0f,
      };
    }

    if (!render_pass->ColorAttachmentReferences().empty()) {
      pipeline_color_blend_attachment_states.resize(
          render_pass->ColorAttachmentReferences().size());
      for (size_t i = 0; i < render_pass->ColorAttachmentReferences().size();
           i++) {
        pipeline_color_blend_attachment_states[i].blendEnable = VK_FALSE;
        pipeline_color_blend_attachment_states[i].srcColorBlendFactor =
            VK_BLEND_FACTOR_ONE;
        pipeline_color_blend_attachment_states[i].dstColorBlendFactor =
            VK_BLEND_FACTOR_ZERO;
        pipeline_color_blend_attachment_states[i].colorBlendOp =
            VK_BLEND_OP_ADD;
        pipeline_color_blend_attachment_states[i].srcAlphaBlendFactor =
            VK_BLEND_FACTOR_ONE;
        pipeline_color_blend_attachment_states[i].dstAlphaBlendFactor =
            VK_BLEND_FACTOR_ZERO;
        pipeline_color_blend_attachment_states[i].alphaBlendOp =
            VK_BLEND_OP_ADD;
        pipeline_color_blend_attachment_states[i].colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      }
    }

    if (!render_pass->ResolveAttachmentReferences().empty()) {
      multisample_state_create_info.alphaToCoverageEnable = VK_TRUE;
      multisample_state_create_info.alphaToOneEnable = VK_TRUE;
    }
  }
}

void PipelineSettings::AddInputBinding(uint32_t binding,
                                       uint32_t stride,
                                       VkVertexInputRate input_rate) {
  vertex_input_binding_descriptions.push_back(VkVertexInputBindingDescription{
      binding,
      stride,
      input_rate,
  });
}

void PipelineSettings::AddInputAttribute(uint32_t binding,
                                         uint32_t location,
                                         VkFormat format,
                                         uint32_t offset) {
  vertex_input_attribute_descriptions.push_back(
      VkVertexInputAttributeDescription{
          location,
          binding,
          format,
          offset,
      });
}

void PipelineSettings::AddShaderStage(ShaderModule *shader_module,
                                      VkShaderStageFlagBits stage) {
  shader_stage_create_infos.push_back(VkPipelineShaderStageCreateInfo{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      nullptr,
      0,
      stage,
      shader_module->Handle(),
      "main",
      nullptr,
  });
}

void PipelineSettings::SetPrimitiveTopology(VkPrimitiveTopology topology) {
  input_assembly_state_create_info.topology = topology;
}

void PipelineSettings::SetMultiSampleState(VkSampleCountFlagBits sample_count) {
  multisample_state_create_info.rasterizationSamples = sample_count;
}

void PipelineSettings::SetCullMode(VkCullModeFlags cull_mode) {
  rasterization_state_create_info.cullMode = cull_mode;
}

}  // namespace grassland::vulkan
