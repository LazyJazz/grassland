#pragma once
#include <optional>

#include "grassland/vulkan/core/core.h"

namespace grassland::vulkan {
class RenderPass {
 public:
  RenderPass(
      class Core *core,
      const std::vector<VkAttachmentDescription> &attachment_descriptions,
      const std::vector<VkAttachmentReference> &color_attachment_references,
      const std::optional<VkAttachmentReference> &depth_attachment_reference =
          std::nullopt,
      const std::vector<VkAttachmentReference> &resolve_attachment_references =
          {});
  ~RenderPass();

  [[nodiscard]] VkRenderPass Handle() const;
  [[nodiscard]] class Core *Core() const;

  [[nodiscard]] const std::vector<VkAttachmentDescription>
      &AttachmentDescriptions() const {
    return attachment_descriptions_;
  }
  [[nodiscard]] const std::vector<VkAttachmentReference>
      &ColorAttachmentReferences() const {
    return color_attachment_references_;
  }
  [[nodiscard]] const std::optional<VkAttachmentReference>
      &DepthAttachmentReference() const {
    return depth_attachment_reference_;
  }
  [[nodiscard]] const std::vector<VkAttachmentReference>
      &ResolveAttachmentReferences() const {
    return resolve_attachment_references_;
  }

 private:
  class Core *core_{};
  VkRenderPass render_pass_{};
  std::vector<VkAttachmentDescription> attachment_descriptions_{};
  std::vector<VkAttachmentReference> color_attachment_references_{};
  std::optional<VkAttachmentReference> depth_attachment_reference_{};
  std::vector<VkAttachmentReference> resolve_attachment_references_{};
};
}  // namespace grassland::vulkan
