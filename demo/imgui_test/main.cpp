#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include "grassland/grassland.h"
#include "imgui.h"

int main() {
  grassland::vulkan::framework::CoreSettings core_settings;
  std::unique_ptr<grassland::vulkan::framework::Core> core =
      std::make_unique<grassland::vulkan::framework::Core>(core_settings);

  VkDescriptorPoolSize pool_sizes[] = {
      {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};

  VkDescriptorPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets = 1000;
  pool_info.poolSizeCount = std::size(pool_sizes);
  pool_info.pPoolSizes = pool_sizes;

  VkDescriptorPool imguiPool;
  vkCreateDescriptorPool(core->GetDevice()->GetHandle(), &pool_info, nullptr,
                         &imguiPool);

  // 2: initialize imgui library

  // this initializes the core structures of imgui
  ImGui::CreateContext();
  ImGui::StyleColorsClassic();
  ImGui_ImplGlfw_InitForVulkan(core->GetWindow(), true);

  // this initializes imgui for Vulkan
  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.Instance = core->GetInstance()->GetHandle();
  init_info.PhysicalDevice = core->GetPhysicalDevice()->GetHandle();
  init_info.Device = core->GetDevice()->GetHandle();
  init_info.Queue = core->GetDevice()->GetGraphicsQueue()->GetHandle();
  init_info.DescriptorPool = imguiPool;
  init_info.MinImageCount = 3;
  init_info.ImageCount = 3;
  init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

  std::unique_ptr<grassland::vulkan::RenderPass> render_pass =
      std::make_unique<grassland::vulkan::RenderPass>(core->GetDevice(),
                                                      VK_FORMAT_B8G8R8A8_UNORM);
  ImGui_ImplVulkan_Init(&init_info, render_pass->GetHandle());

  // execute a gpu command to upload imgui font textures
  grassland::vulkan::helper::SingleTimeCommands(
      core->GetCommandPool(),
      [&](VkCommandBuffer cmd) { ImGui_ImplVulkan_CreateFontsTexture(cmd); });

  // clear font textures from cpu data
  ImGui_ImplVulkan_DestroyFontUploadObjects();

  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

  auto paint_buffer =
      std::make_unique<grassland::vulkan::framework::TextureImage>(
          core.get(), core->GetFramebufferWidth(), core->GetFramebufferHeight(),
          VK_FORMAT_B8G8R8A8_UNORM,
          VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);

  auto framebuffer = std::make_unique<grassland::vulkan::Framebuffer>(
      core->GetDevice(), core->GetFramebufferWidth(),
      core->GetFramebufferHeight(), render_pass.get(),
      paint_buffer->GetImageView());
  core->SetFrameSizeCallback([&core, &render_pass, &framebuffer, &paint_buffer](
                                 int width, int height) {
    paint_buffer->Resize(width, height);
    framebuffer = std::make_unique<grassland::vulkan::Framebuffer>(
        core->GetDevice(), width, height, render_pass.get(),
        paint_buffer->GetImageView());
  });

  while (!glfwWindowShouldClose(core->GetWindow())) {
    ImGui_ImplGlfw_NewFrame();
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();
    ImGui::Begin("Hello, World!");
    ImGui::Text("Hello, I'm GUI.");
    ImGui::End();
    ImGui::Render();
    core->BeginCommandRecord();
    paint_buffer->ClearColor({0.6, 0.7, 0.8, 1.0});
    auto command_buffer = core->GetCommandBuffer()->GetHandle();
    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = render_pass->GetHandle();
    renderPassInfo.framebuffer = framebuffer->GetHandle();
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = framebuffer->GetExtent();
    renderPassInfo.clearValueCount = 0;
    renderPassInfo.pClearValues = nullptr;

    vkCmdBeginRenderPass(command_buffer, &renderPassInfo,
                         VK_SUBPASS_CONTENTS_INLINE);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(),
                                    core->GetCommandBuffer()->GetHandle());
    vkCmdEndRenderPass(core->GetCommandBuffer()->GetHandle());
    core->Output(paint_buffer.get());
    core->EndCommandRecordAndSubmit();
    glfwPollEvents();
  }
  core->GetDevice()->WaitIdle();
  framebuffer.reset();
  paint_buffer.reset();

  vkDestroyDescriptorPool(core->GetDevice()->GetHandle(), imguiPool, nullptr);
  ImGui_ImplVulkan_Shutdown();
}
