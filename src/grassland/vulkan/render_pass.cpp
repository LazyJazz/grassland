#include "render_pass.h"

namespace grassland::vulkan {
RenderPass::RenderPass(const class Device &device,
                       VkFormat color_format,
                       VkFormat depth_format)
    : RenderPass(device, std::vector<VkFormat>{color_format}, depth_format) {
}

RenderPass::RenderPass(const class grassland::vulkan::Device &device,
                       const std::vector<VkFormat> &color_formats,
                       VkFormat depth_format)
    : device_(device) {
  std::vector<VkAttachmentDescription> attachments;
  std::vector<VkAttachmentReference> color_references;
  VkAttachmentReference depth_reference{};
  VkSubpassDescription subpass_desc{};
  subpass_desc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

  for (auto color_format : color_formats) {
    VkAttachmentDescription color_attachment{};
    color_attachment.format = color_format;
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    attachments.push_back(color_attachment);

    VkAttachmentReference color_reference{};
    color_reference.attachment = attachments.size() - 1;
    color_reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  }

  subpass_desc.colorAttachmentCount = color_references.size();
  subpass_desc.pColorAttachments = color_references.data();

  if (depth_format != VK_FORMAT_UNDEFINED) {
    VkAttachmentDescription depth_attachment{};
    depth_attachment.format = depth_format;
    depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attachment.initialLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth_attachment.finalLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    attachments.push_back(depth_attachment);
    depth_reference.attachment = attachments.size() - 1;
    depth_reference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    subpass_desc.pDepthStencilAttachment = &depth_reference;
  }

  VkSubpassDependency dependency{};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask = 0;

  VkRenderPassCreateInfo pass_info{};
  pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  pass_info.attachmentCount = attachments.size();
  pass_info.pAttachments = attachments.data();
  pass_info.dependencyCount = 1;
  pass_info.pDependencies = &dependency;
  pass_info.subpassCount = 1;
  pass_info.pSubpasses = &subpass_desc;
  GRASSLAND_VULKAN_CHECK(
      vkCreateRenderPass(device_.Handle(), &pass_info, nullptr, &render_pass_));
}

RenderPass::~RenderPass() {
  if (render_pass_ != VK_NULL_HANDLE) {
    vkDestroyRenderPass(device_.Handle(), render_pass_, nullptr);
  }
}

}  // namespace grassland::vulkan
