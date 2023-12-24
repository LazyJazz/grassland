#pragma once
#include <optional>

#include "grassland/vulkan/core/core.h"

namespace grassland::vulkan {

struct SubpassSettings {
  std::vector<VkAttachmentReference> input_attachment_references{};
  std::vector<VkAttachmentReference> color_attachment_references{};
  std::optional<VkAttachmentReference> depth_attachment_reference{};
  std::vector<VkAttachmentReference> resolve_attachment_references{};
  std::vector<uint32_t> preserve_attachment_references{};

  SubpassSettings() = default;
  SubpassSettings(
      const std::vector<VkAttachmentReference> &color_attachment_references,
      const std::optional<VkAttachmentReference> &depth_attachment_reference =
          std::nullopt,
      const std::vector<VkAttachmentReference> &resolve_attachment_references =
          {});
  SubpassSettings(
      const std::vector<VkAttachmentReference> &input_attachment_references,
      const std::vector<VkAttachmentReference> &color_attachment_references,
      const std::optional<VkAttachmentReference> &depth_attachment_reference =
          std::nullopt,
      const std::vector<VkAttachmentReference> &resolve_attachment_references =
          {},
      const std::vector<uint32_t> &preserve_attachment_references = {});

  VkSubpassDescription Description() const;

  const std::vector<VkAttachmentReference> &InputAttachmentReferences() const {
    return input_attachment_references;
  }

  const std::vector<VkAttachmentReference> &ColorAttachmentReferences() const {
    return color_attachment_references;
  }

  const std::optional<VkAttachmentReference> &DepthAttachmentReference() const {
    return depth_attachment_reference;
  }

  const std::vector<VkAttachmentReference> &ResolveAttachmentReferences()
      const {
    return resolve_attachment_references;
  }

  const std::vector<uint32_t> &PreserveAttachmentReferences() const {
    return preserve_attachment_references;
  }
};

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
  RenderPass(
      class Core *core,
      const std::vector<VkAttachmentDescription> &attachment_descriptions,
      const std::vector<struct SubpassSettings> &subpass_settings,
      const std::vector<VkSubpassDependency> &dependencies);
  ~RenderPass();

  [[nodiscard]] VkRenderPass Handle() const;
  [[nodiscard]] class Core *Core() const;

  const std::vector<VkAttachmentDescription> &AttachmentDescriptions() const {
    return attachment_descriptions_;
  }

  const std::vector<struct SubpassSettings> &SubpassSettings() const {
    return subpass_settings_;
  }

 private:
  class Core *core_{};
  VkRenderPass render_pass_{};
  std::vector<VkAttachmentDescription> attachment_descriptions_{};
  std::vector<struct SubpassSettings> subpass_settings_{};
};
}  // namespace grassland::vulkan
