#include "grassland/vulkan/pipeline/render_pass.h"

namespace grassland::vulkan {

RenderPass::RenderPass(
    class Core *core,
    const std::vector<VkAttachmentDescription> &attachment_descriptions,
    const std::vector<VkAttachmentReference> &color_attachment_references,
    const std::optional<VkAttachmentReference> &depth_attachment_reference,
    const std::vector<VkAttachmentReference> &resolve_attachment_references)
    : RenderPass(core,
                 attachment_descriptions,
                 std::vector<struct SubpassSettings>{
                     {color_attachment_references, depth_attachment_reference,
                      resolve_attachment_references}},
                 std::vector<VkSubpassDependency>{
                     {VK_SUBPASS_EXTERNAL, 0,
                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0}}) {
}

RenderPass::RenderPass(
    struct Core *core,
    const std::vector<VkAttachmentDescription> &attachment_descriptions,
    const std::vector<struct SubpassSettings> &subpass_settings,
    const std::vector<VkSubpassDependency> &dependencies)
    : core_(core),
      attachment_descriptions_(attachment_descriptions),
      subpass_settings_(subpass_settings) {
  std::vector<VkSubpassDescription> subpass_descriptions{};
  for (const auto &subpass_setting : subpass_settings) {
    subpass_descriptions.push_back(subpass_setting.Description());
  }

  // Build RenderPass here
  VkRenderPassCreateInfo render_pass_create_info{};
  render_pass_create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  render_pass_create_info.attachmentCount = attachment_descriptions_.size();
  render_pass_create_info.pAttachments = attachment_descriptions_.data();
  render_pass_create_info.subpassCount = subpass_descriptions.size();
  render_pass_create_info.pSubpasses = subpass_descriptions.data();
  render_pass_create_info.dependencyCount = dependencies.size();
  render_pass_create_info.pDependencies = dependencies.data();

  if (vkCreateRenderPass(core_->Device()->Handle(), &render_pass_create_info,
                         nullptr, &render_pass_) != VK_SUCCESS) {
    LAND_ERROR("[Vulkan] failed to create render pass!");
  }
}

RenderPass::~RenderPass() {
  vkDestroyRenderPass(core_->Device()->Handle(), render_pass_, nullptr);
}

VkRenderPass RenderPass::Handle() const {
  return render_pass_;
}

class Core *RenderPass::Core() const {
  return core_;
}

VkSubpassDescription SubpassSettings::Description() const {
  VkSubpassDescription description{};
  description.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  description.inputAttachmentCount = input_attachment_references.size();
  description.pInputAttachments = input_attachment_references.data();
  description.colorAttachmentCount = color_attachment_references.size();
  description.pColorAttachments = color_attachment_references.data();
  if (depth_attachment_reference.has_value()) {
    description.pDepthStencilAttachment = &depth_attachment_reference.value();
  }
  if (!resolve_attachment_references.empty()) {
    description.pResolveAttachments = resolve_attachment_references.data();
  }
  description.preserveAttachmentCount = preserve_attachment_references.size();
  if (description.preserveAttachmentCount) {
    description.pPreserveAttachments = preserve_attachment_references.data();
  }
  return description;
}

SubpassSettings::SubpassSettings(
    const std::vector<VkAttachmentReference> &color_attachment_references,
    const std::optional<VkAttachmentReference> &depth_attachment_reference,
    const std::vector<VkAttachmentReference> &resolve_attachment_references)
    : color_attachment_references(color_attachment_references),
      depth_attachment_reference(depth_attachment_reference),
      resolve_attachment_references(resolve_attachment_references) {
}

SubpassSettings::SubpassSettings(
    const std::vector<VkAttachmentReference> &input_attachment_references,
    const std::vector<VkAttachmentReference> &color_attachment_references,
    const std::optional<VkAttachmentReference> &depth_attachment_reference,
    const std::vector<VkAttachmentReference> &resolve_attachment_references,
    const std::vector<uint32_t> &preserve_attachment_references)
    : input_attachment_references(input_attachment_references),
      color_attachment_references(color_attachment_references),
      depth_attachment_reference(depth_attachment_reference),
      resolve_attachment_references(resolve_attachment_references),
      preserve_attachment_references(preserve_attachment_references) {
}

}  // namespace grassland::vulkan
