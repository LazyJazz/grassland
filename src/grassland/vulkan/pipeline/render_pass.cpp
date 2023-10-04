#include "grassland/vulkan/pipeline/render_pass.h"

namespace grassland::vulkan {

RenderPass::RenderPass(
    class Core *core,
    const std::vector<VkAttachmentDescription> &attachment_descriptions,
    const std::vector<VkAttachmentReference> &color_attachment_references,
    const std::optional<VkAttachmentReference> &depth_attachment_reference,
    const std::vector<VkAttachmentReference> &resolve_attachment_references)
    : core_(core),
      attachment_descriptions_(attachment_descriptions),
      color_attachment_references_(color_attachment_references),
      depth_attachment_reference_(depth_attachment_reference),
      resolve_attachment_references_(resolve_attachment_references) {
  // Build RenderPass here

  VkSubpassDescription subpass_description{};
  subpass_description.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass_description.colorAttachmentCount =
      color_attachment_references_.size();
  subpass_description.pColorAttachments = color_attachment_references_.data();
  if (depth_attachment_reference_.has_value()) {
    subpass_description.pDepthStencilAttachment =
        &depth_attachment_reference_.value();
  }
  if (!resolve_attachment_references_.empty()) {
    subpass_description.pResolveAttachments =
        resolve_attachment_references_.data();
  }

  VkSubpassDependency subpass_dependency{};
  subpass_dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  subpass_dependency.dstSubpass = 0;
  subpass_dependency.srcStageMask =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  subpass_dependency.srcAccessMask = 0;
  subpass_dependency.dstStageMask =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  subpass_dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  subpass_dependency.dependencyFlags = 0;

  VkRenderPassCreateInfo render_pass_create_info{};
  render_pass_create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  render_pass_create_info.attachmentCount = attachment_descriptions_.size();
  render_pass_create_info.pAttachments = attachment_descriptions_.data();
  render_pass_create_info.subpassCount = 1;
  render_pass_create_info.pSubpasses = &subpass_description;
  render_pass_create_info.dependencyCount = 1;
  render_pass_create_info.pDependencies = &subpass_dependency;

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

}  // namespace grassland::vulkan
