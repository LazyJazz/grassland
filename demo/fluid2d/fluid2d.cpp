#include "fluid2d.h"

#include "glm/gtc/matrix_transform.hpp"
#include "iomanip"
#include "iostream"
#include "random"

Fluid2D::Fluid2D(int n_particles) {
  vulkan_legacy::framework::CoreSettings core_settings;
  core_settings.window_title = "Fluid2D";
  core_settings.window_width = 512;
  core_settings.window_height = 1024;
  core = std::make_unique<vulkan_legacy::framework::Core>(core_settings);
  particles.resize(n_particles);
}

void Fluid2D::Run() {
  OnInit();
  while (!glfwWindowShouldClose(core->GetWindow())) {
    OnLoop();
    glfwPollEvents();
  }
  core->GetDevice()->WaitIdle();
  OnClose();
}

void Fluid2D::OnInit() {
  texture_image = std::make_unique<vulkan_legacy::framework::TextureImage>(
      core.get(), 512, 1024, VK_FORMAT_R32G32B32A32_SFLOAT);
  sampler = std::make_unique<vulkan_legacy::Sampler>(core->GetDevice(),
                                                     VK_FILTER_LINEAR);
  framebuffer = std::make_unique<vulkan_legacy::framework::TextureImage>(
      core.get(), core->GetFramebufferWidth(), core->GetFramebufferHeight(),
      VK_FORMAT_B8G8R8A8_UNORM);
  bkground_render_node =
      std::make_unique<vulkan_legacy::framework::RenderNode>(core.get());
  bkground_render_node->AddColorAttachment(framebuffer.get());
  bkground_render_node->AddShader("bkground.vert.spv",
                                  VK_SHADER_STAGE_VERTEX_BIT);
  bkground_render_node->AddShader("bkground.frag.spv",
                                  VK_SHADER_STAGE_FRAGMENT_BIT);
  bkground_render_node->AddUniformBinding(texture_image.get(), sampler.get(),
                                          VK_SHADER_STAGE_FRAGMENT_BIT);
  bkground_render_node->VertexInput({VK_FORMAT_R32G32_SFLOAT});
  bkground_render_node->BuildRenderNode();

  glm::vec2 vertices[] = {
      {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
  uint32_t indices[] = {0, 1, 2, 1, 2, 3};
  rectangle_vertex_buffer =
      std::make_unique<vulkan_legacy::framework::StaticBuffer<glm::vec2>>(
          core.get(), 4);
  rectangle_index_buffer =
      std::make_unique<vulkan_legacy::framework::StaticBuffer<uint32_t>>(
          core.get(), 6);
  rectangle_vertex_buffer->Upload(vertices);
  rectangle_index_buffer->Upload(indices);

  core->SetFrameSizeCallback([this](int width, int height) {
    if (render_node) {
      render_node->BuildRenderNode(width, height);
    }
    if (bkground_render_node) {
      bkground_render_node->BuildRenderNode(width, height);
    }
    framebuffer->Resize(width, height);
    glm::mat4 global_transform{
        2.0f / width, 0.0f, 0.0f, 0.0f, 0.0f,  2.0f / height, 0.0f, 0.0f,
        0.0f,         0.0f, 1.0f, 0.0f, -1.0f, -1.0f,         0.0f, 1.0f};
    global_transform_buffer->Upload(&global_transform);
  });

  for (int x = 0; x < 512; x++) {
    for (int y = 0; y < 1024; y++) {
      glm::vec2 pix_pos{x + 0.5f, y + 0.5f};
      pix_pos *= 20.0f / 512.0f;
      if (InsideContainer(pix_pos)) {
        bkground_image[y][x] = glm::vec4{glm::vec3{0.0f}, 1.0f};
      } else {
        bkground_image[y][x] = glm::vec4{glm::vec3{1.0f}, 1.0f};
      }
    }
  }
  vulkan_legacy::framework::StaticBuffer<glm::vec4> upload_buffer(core.get(),
                                                                  512 * 1024);
  upload_buffer.Upload(reinterpret_cast<const glm::vec4 *>(bkground_image));
  vulkan_legacy::UploadImage(core->GetCommandPool(), texture_image->GetImage(),
                             upload_buffer.GetBuffer(0));

  {
    std::vector<glm::vec2> circle_vertices;
    std::vector<uint32_t> circle_indices;
    const int precision = 12;
    const auto pi = glm::pi<float>();
    const float inv_precision = 1.0f / precision;
    for (int i = 0; i < precision; i++) {
      float theta = i * 2.0f * pi * inv_precision;
      circle_vertices.emplace_back(std::sin(theta), std::cos(theta));
      circle_indices.push_back(i);
      circle_indices.push_back((i + 1) % precision);
      circle_indices.push_back(precision);
    }
    circle_vertices.emplace_back(0.0f, 0.0f);
    vertex_buffer =
        std::make_unique<vulkan_legacy::framework::StaticBuffer<glm::vec2>>(
            core.get(), circle_vertices.size());
    index_buffer =
        std::make_unique<vulkan_legacy::framework::StaticBuffer<uint32_t>>(
            core.get(), circle_indices.size());
    vertex_buffer->Upload(circle_vertices.data());
    index_buffer->Upload(circle_indices.data());
  }

  render_objects = std::make_unique<
      vulkan_legacy::framework::DynamicBuffer<RenderObjectInfo>>(core.get(),
                                                                 65536);

  global_transform_buffer =
      std::make_unique<vulkan_legacy::framework::StaticBuffer<glm::mat4>>(
          core.get(), 1);
  glm::mat4 global_transform{2.0f / core->GetFramebufferWidth(),
                             0.0f,
                             0.0f,
                             0.0f,
                             0.0f,
                             2.0f / core->GetFramebufferHeight(),
                             0.0f,
                             0.0f,
                             0.0f,
                             0.0f,
                             1.0f,
                             0.0f,
                             -1.0f,
                             -1.0f,
                             0.0f,
                             1.0f};
  global_transform_buffer->Upload(&global_transform);
  render_node =
      std::make_unique<vulkan_legacy::framework::RenderNode>(core.get());
  render_node->AddColorAttachment(framebuffer.get());
  render_node->AddUniformBinding(global_transform_buffer.get(),
                                 VK_SHADER_STAGE_VERTEX_BIT);
  render_node->AddBufferBinding(render_objects.get(),
                                VK_SHADER_STAGE_VERTEX_BIT);
  render_node->AddShader("particle.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
  render_node->AddShader("particle.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
  render_node->VertexInput({VK_FORMAT_R32G32_SFLOAT});
  render_node->BuildRenderNode();

  InitParticles();
}

void Fluid2D::OnLoop() {
  OnUpdate();
  OnRender();
}

void Fluid2D::OnClose() {
}

bool Fluid2D::InsideContainer(glm::vec2 pos) {
  return glm::length(pos - glm::vec2{10.0f, 12.0f}) < 8.0f ||
         glm::length(pos - glm::vec2{10.0f, 28.0f}) < 8.0f ||
         (pos.x >= 7.0f && pos.x <= 13.0f && pos.y > 12.0f && pos.y < 28.0f);
}

void Fluid2D::OnUpdate() {
  SolveParticleDynamics();
  render_object_infos.clear();
  ComposeScene();
  UpdateRenderAssets();
}

void Fluid2D::OnRender() {
  core->BeginCommandRecord();
  framebuffer->ClearColor({0.6f, 0.7f, 0.8f, 1.0f});
  bkground_render_node->Draw(rectangle_vertex_buffer.get(),
                             rectangle_index_buffer.get(), 6, 0, 1);
  render_node->Draw(vertex_buffer.get(), index_buffer.get(),
                    index_buffer->Size(), 0, render_object_infos.size());
  core->Output(framebuffer.get());
  core->EndCommandRecordAndSubmit();
}

void Fluid2D::UpdateRenderAssets() {
  std::memcpy(render_objects->Map(), render_object_infos.data(),
              render_object_infos.size() * sizeof(RenderObjectInfo));
}

void Fluid2D::DrawCircle(glm::vec2 origin, float radius, glm::vec4 color) {
  render_object_infos.push_back(
      {glm::mat4{radius, 0.0f, 0.0f, 0.0f, 0.0f, radius, 0.0f, 0.0f, 0.0f, 0.0f,
                 1.0f, 0.0f, origin.x, origin.y, 0.0f, 1.0f},
       color});
}

void Fluid2D::ComposeScene() {
  glm::vec2 transform =
      glm::vec2{core->GetFramebufferWidth(), core->GetFramebufferHeight()} /
      glm::vec2{GRID_SIZE_X * DELTA_X, GRID_SIZE_Y * DELTA_X};
  for (auto particle : particles) {
    DrawCircle(particle.position * transform, 2.0f,
               particle.type == TYPE_AIR ? glm::vec4{1.0f, 0.0f, 0.0f, 1.0f}
                                         : glm::vec4{0.0f, 0.0f, 1.0f, 1.0f});
  }
}

void Fluid2D::InitParticles() {
  std::mt19937_64 rd{};
  for (auto &particle : particles) {
    glm::vec2 pos;
    do {
      pos = {std::uniform_real_distribution<float>(0.0f,
                                                   GRID_SIZE_X * DELTA_X)(rd),
             std::uniform_real_distribution<float>(0.0f,
                                                   GRID_SIZE_Y * DELTA_X)(rd)};
    } while (!InsideContainer(pos));
    particle.position = pos;
    particle.velocity = {};
    if (particle.position.y < 0.5f * DELTA_X * GRID_SIZE_Y) {
      particle.type = TYPE_AIR;
    } else {
      particle.type = TYPE_LIQ;
    }
  }
}

void Fluid2D::SolveParticleDynamics() {
  for (auto &particle : particles) {
    particle.velocity += GRAVITY * delta_t;
  }

  std::memset(u_grid, 0, sizeof(u_grid));
  std::memset(v_grid, 0, sizeof(v_grid));
  std::memset(u_weight_grid, 0, sizeof(u_weight_grid));
  std::memset(v_weight_grid, 0, sizeof(v_weight_grid));
  std::memset(level_set, 0, sizeof(level_set));

  auto kernel_func = [](glm::vec2 v) {
    auto len = glm::length(v);
    if (len < 0.5f) {
      return 0.75f - len * len;
    } else if (len < 1.5f) {
      return 0.5f * (1.5f - len) * (1.5f - len);
    }
    return 0.0f;
  };

  for (auto particle : particles) {
    auto u_pos =
        glm::vec2{u_grid_transform * glm::vec3{particle.position, 1.0f}};
    auto v_pos =
        glm::vec2{v_grid_transform * glm::vec3{particle.position, 1.0f}};
    auto c_pos =
        glm::vec2{c_grid_transform * glm::vec3{particle.position, 1.0f}};
    glm::ivec2 u_ipos{u_pos + 0.5f};
    glm::ivec2 v_ipos{v_pos + 0.5f};
    glm::ivec2 c_ipos{c_pos + 0.5f};
    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        glm::ivec2 offset{i, j};
        auto u_opos = u_ipos + offset;
        auto v_opos = v_ipos + offset;
        auto c_opos = c_ipos + offset;
        if (u_opos.x >= 0 && u_opos.x < GRID_SIZE_X + 1 && u_opos.y >= 0 &&
            u_opos.y < GRID_SIZE_Y) {
          auto weight = kernel_func(u_pos - glm::vec2{u_opos});
          u_weight_grid[particle.type][u_opos.x][u_opos.y] += weight;
          u_grid[particle.type][u_opos.x][u_opos.y] +=
              weight * particle.velocity.x;
        }
        if (v_opos.x >= 0 && v_opos.x < GRID_SIZE_X && v_opos.y >= 0 &&
            v_opos.y < GRID_SIZE_Y + 1) {
          auto weight = kernel_func(v_pos - glm::vec2{v_opos});
          v_weight_grid[particle.type][v_opos.x][v_opos.y] += weight;
          v_grid[particle.type][v_opos.x][v_opos.y] +=
              weight * particle.velocity.y;
        }
        //        if (c_opos.x >= 0 && c_opos.x < GRID_SIZE_X + 1 && c_opos.y >=
        //        0 && c_opos.y < GRID_SIZE_Y + 1) {
        //
        //        }
      }
    }
  }

  for (int t = 0; t < 2; t++) {
    for (int x = 0; x < GRID_SIZE_X + 1; x++) {
      for (int y = 0; y < GRID_SIZE_Y; y++) {
        float &v = u_grid[t][x][y];
        float &w = u_weight_grid[t][x][y];
        if (w > 1e-5f) {
          v /= w;
        }
      }
    }
  }

  for (int t = 0; t < 2; t++) {
    for (int x = 0; x < GRID_SIZE_X; x++) {
      for (int y = 0; y < GRID_SIZE_Y + 1; y++) {
        float &v = v_grid[t][x][y];
        float &w = v_weight_grid[t][x][y];
        if (w > 1e-5f) {
          v /= w;
        }
      }
    }
  }

  for (int x = 0; x < GRID_SIZE_X + 1; x++) {
    for (int y = 0; y < GRID_SIZE_Y + 1; y++) {
      glm::vec2 grid_point_pos{x * DELTA_X, y * DELTA_X};
      float nearest_air = 1e10;
      float nearest_liq = 1e10;
      for (auto particle : particles) {
        if (particle.type == TYPE_AIR) {
          nearest_air = std::min(
              nearest_air, glm::length(particle.position - grid_point_pos));
        } else {
          nearest_liq = std::min(
              nearest_liq, glm::length(particle.position - grid_point_pos));
        }
      }
      level_set[x][y] = (nearest_air - nearest_liq) * 0.5f;
    }
  }

  {
    std::memset(u_weight_grid, 0, sizeof(u_weight_grid));
    std::memset(v_weight_grid, 0, sizeof(v_weight_grid));
    const int precision = 16;
    const float inv_precision = 1.0f / (float)precision;

    for (int x = 0; x < GRID_SIZE_X + 1; x++) {
      for (int y = 0; y < GRID_SIZE_Y; y++) {
        glm::vec2 grid_point_pos{x * DELTA_X, y * DELTA_X};
        for (int k = 0; k < precision; k++) {
          float rel_scale = (k + 0.5f) * inv_precision;
          auto sample_point =
              grid_point_pos + glm::vec2{0.0f, rel_scale * DELTA_X};
          if (InsideContainer(sample_point)) {
            auto level_set_inter = level_set[x][y] * (1.0f - rel_scale) +
                                   level_set[x][y + 1] * rel_scale;
            if (level_set_inter > 0.0f) {
              u_weight_grid[TYPE_LIQ][x][y] += inv_precision;
            } else {
              u_weight_grid[TYPE_AIR][x][y] += inv_precision;
            }
          }
        }
      }
    }

    for (int x = 0; x < GRID_SIZE_X; x++) {
      for (int y = 0; y < GRID_SIZE_Y + 1; y++) {
        glm::vec2 grid_point_pos{x * DELTA_X, y * DELTA_X};
        for (int k = 0; k < precision; k++) {
          float rel_scale = (k + 0.5f) * inv_precision;
          auto sample_point =
              grid_point_pos + glm::vec2{rel_scale * DELTA_X, 0.0f};
          if (InsideContainer(sample_point)) {
            auto level_set_inter = level_set[x][y] * (1.0f - rel_scale) +
                                   level_set[x + 1][y] * rel_scale;
            if (level_set_inter > 0.0f) {
              v_weight_grid[TYPE_LIQ][x][y] += inv_precision;
            } else {
              v_weight_grid[TYPE_AIR][x][y] += inv_precision;
            }
          }
        }
      }
    }
  }

  std::cout << std::fixed << std::setprecision(7);
  std::memset(divergence, 0, sizeof(divergence));
  for (int y = 0; y < GRID_SIZE_Y; y++) {
    for (int x = 0; x < GRID_SIZE_X; x++) {
      int id = y * GRID_SIZE_X + x;
      divergence[id] += u_weight_grid[0][x][y] * u_grid[0][x][y] +
                        u_weight_grid[1][x][y] * u_grid[1][x][y];
      divergence[id] += v_weight_grid[0][x][y] * v_grid[0][x][y] +
                        v_weight_grid[1][x][y] * v_grid[1][x][y];
      divergence[id] -= u_weight_grid[0][x + 1][y] * u_grid[0][x + 1][y] +
                        u_weight_grid[1][x + 1][y] * u_grid[1][x + 1][y];
      divergence[id] -= v_weight_grid[0][x][y + 1] * v_grid[0][x][y + 1] +
                        v_weight_grid[1][x][y + 1] * v_grid[1][x][y + 1];
    }
  }

  PrintAMatrix();
  SolvePressure();
  //    SolvePressureImpactToDivergence(pressure, buffer);
  //    for (int i = 0; i < GRID_SIZE_X * GRID_SIZE_Y; i++) {
  //      std::cout << buffer[i] << ' ' << divergence[i] << std::endl;
  //    }
  // PrintMatrixAndOpen(pressure);
  //  PrintMatrixAndOpen(buffer);

  for (int y = 0; y < GRID_SIZE_Y; y++) {
    for (int x = 1; x < GRID_SIZE_X; x++) {
      int id = y * GRID_SIZE_X + x;
      u_grid[TYPE_AIR][x][y] =
          (pressure[id - 1] - pressure[id]) / (RHO_AIR * DELTA_X) * delta_t;
      u_grid[TYPE_LIQ][x][y] =
          (pressure[id - 1] - pressure[id]) / (RHO_LIQ * DELTA_X) * delta_t;
    }
  }

  for (int y = 1; y < GRID_SIZE_Y; y++) {
    for (int x = 0; x < GRID_SIZE_X; x++) {
      int id = y * GRID_SIZE_X + x;
      v_grid[TYPE_AIR][x][y] = (pressure[id - GRID_SIZE_X] - pressure[id]) /
                               (RHO_AIR * DELTA_X) * delta_t;
      v_grid[TYPE_LIQ][x][y] = (pressure[id - GRID_SIZE_X] - pressure[id]) /
                               (RHO_LIQ * DELTA_X) * delta_t;
    }
  }

  for (auto &particle : particles) {
    auto u_pos =
        glm::vec2{u_grid_transform * glm::vec3{particle.position, 1.0f}};
    auto v_pos =
        glm::vec2{v_grid_transform * glm::vec3{particle.position, 1.0f}};
    glm::ivec2 u_ipos{u_pos + 0.5f};
    glm::ivec2 v_ipos{v_pos + 0.5f};
    glm::vec2 vel_change{}, vel_weight{};
    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        glm::ivec2 offset{i, j};
        auto u_opos = u_ipos + offset;
        auto v_opos = v_ipos + offset;
        if (u_opos.x >= 0 && u_opos.x < GRID_SIZE_X + 1 && u_opos.y >= 0 &&
            u_opos.y < GRID_SIZE_Y &&
            u_weight_grid[particle.type][u_opos.x][u_opos.y] > 1e-4f) {
          auto weight = kernel_func(u_pos - glm::vec2{u_opos});
          vel_weight.x += weight;
          vel_change.x += weight * u_grid[particle.type][u_opos.x][u_opos.y];
        }
        if (v_opos.x >= 0 && v_opos.x < GRID_SIZE_X && v_opos.y >= 0 &&
            v_opos.y < GRID_SIZE_Y + 1 &&
            v_weight_grid[particle.type][v_opos.x][v_opos.y] > 1e-4f) {
          auto weight = kernel_func(v_pos - glm::vec2{v_opos});
          vel_weight.y += weight;
          vel_change.y += weight * v_grid[particle.type][v_opos.x][v_opos.y];
        }
      }
    }
    if (std::abs(vel_weight.x) > 1e-4f) {
      vel_change.x /= vel_weight.x;
    }
    if (std::abs(vel_weight.y) > 1e-4f) {
      vel_change.y /= vel_weight.y;
    }
    particle.velocity += vel_change;
  }

  //      std::cout <<
  //          "========================================================================================================\n";
  //      std::cout << std::fixed << std::setprecision(7);
  //      for (int y = 0; y < GRID_SIZE_Y + 1; y++) {
  //        for (int x = 0; x < GRID_SIZE_X; x++) {
  //          std::cout << v_grid[0][x][y] << ' ';
  //        }
  //        std::cout << std::endl;
  //      }
  //      std::cout <<
  //          "--------------------------------------------------------------------------------------------------------\n";
  //      for (int y = 0; y < GRID_SIZE_Y + 1; y++) {
  //        for (int x = 0; x < GRID_SIZE_X; x++) {
  //          std::cout << v_grid[1][x][y] << ' ';
  //        }
  //        std::cout << std::endl;
  //      }
  //      std::cout <<
  //          "--------------------------------------------------------------------------------------------------------\n";
  //      std::cout << std::fixed << std::setprecision(7);
  //      for (int y = 0; y < GRID_SIZE_Y; y++) {
  //        for (int x = 0; x < GRID_SIZE_X + 1; x++) {
  //          std::cout << u_grid[0][x][y] << ' ';
  //        }
  //        std::cout << std::endl;
  //      }
  //      std::cout <<
  //          "--------------------------------------------------------------------------------------------------------\n";
  //      for (int y = 0; y < GRID_SIZE_Y; y++) {
  //        for (int x = 0; x < GRID_SIZE_X + 1; x++) {
  //          std::cout << u_grid[1][x][y] << ' ';
  //        }
  //        std::cout << std::endl;
  //      }

  //    std::cout <<
  //        "========================================================================================================\n";
  //    std::cout << std::fixed << std::setprecision(7);
  //    for (int y = 0; y < GRID_SIZE_Y + 1; y++) {
  //      for (int x = 0; x < GRID_SIZE_X; x++) {
  //        std::cout << v_weight_grid[0][x][y] << ' ';
  //      }
  //      std::cout << std::endl;
  //    }
  //    std::cout <<
  //        "--------------------------------------------------------------------------------------------------------\n";
  //    for (int y = 0; y < GRID_SIZE_Y + 1; y++) {
  //      for (int x = 0; x < GRID_SIZE_X; x++) {
  //        std::cout << v_weight_grid[1][x][y] << ' ';
  //      }
  //      std::cout << std::endl;
  //    }
  //    std::cout <<
  //        "--------------------------------------------------------------------------------------------------------\n";
  //    std::cout << std::fixed << std::setprecision(7);
  //    for (int y = 0; y < GRID_SIZE_Y; y++) {
  //      for (int x = 0; x < GRID_SIZE_X + 1; x++) {
  //        std::cout << u_weight_grid[0][x][y] << ' ';
  //      }
  //      std::cout << std::endl;
  //    }
  //    std::cout <<
  //        "--------------------------------------------------------------------------------------------------------\n";
  //    for (int y = 0; y < GRID_SIZE_Y; y++) {
  //      for (int x = 0; x < GRID_SIZE_X + 1; x++) {
  //        std::cout << u_weight_grid[1][x][y] << ' ';
  //      }
  //      std::cout << std::endl;
  //    }
  //    std::cout <<
  //        "--------------------------------------------------------------------------------------------------------\n";
  //    for (int y = 0; y < GRID_SIZE_Y; y++) {
  //      for (int x = 0; x < GRID_SIZE_X; x++) {
  //        std::cout << level_set[x][y] << ' ';
  //      }
  //      std::cout << std::endl;
  //    }

  for (auto &particle : particles) {
    particle.position += particle.velocity * delta_t;
  }
}

void Fluid2D::SolvePressureImpactToDivergence(const float *pressure,
                                              float *delta_divergence) {
  std::memset(delta_divergence, 0, sizeof(float) * GRID_SIZE_X * GRID_SIZE_Y);
  for (int y = 0; y < GRID_SIZE_Y; y++) {
    for (int x = 0; x < GRID_SIZE_X; x++) {
      int id = y * GRID_SIZE_X + x;
      if (x) {
        auto value = pressure[id] * delta_t *
                     (u_weight_grid[TYPE_AIR][x][y] / RHO_AIR +
                      u_weight_grid[TYPE_LIQ][x][y] / RHO_LIQ) /
                     DELTA_X;
        delta_divergence[id] += value;
        delta_divergence[id - 1] -= value;
      }
      if (y) {
        auto value = pressure[id] * delta_t *
                     (v_weight_grid[TYPE_AIR][x][y] / RHO_AIR +
                      v_weight_grid[TYPE_LIQ][x][y] / RHO_LIQ) /
                     DELTA_X;
        delta_divergence[id] += value;
        delta_divergence[id - GRID_SIZE_X] -= value;
      }
      if (x < GRID_SIZE_X - 1) {
        auto value = pressure[id] * delta_t *
                     (u_weight_grid[TYPE_AIR][x + 1][y] / RHO_AIR +
                      u_weight_grid[TYPE_LIQ][x + 1][y] / RHO_LIQ) /
                     DELTA_X;
        delta_divergence[id] += value;
        delta_divergence[id + 1] -= value;
      }
      if (y < GRID_SIZE_Y - 1) {
        auto value = pressure[id] * delta_t *
                     (v_weight_grid[TYPE_AIR][x][y + 1] / RHO_AIR +
                      v_weight_grid[TYPE_LIQ][x][y + 1] / RHO_LIQ) /
                     DELTA_X;
        delta_divergence[id] += value;
        delta_divergence[id + GRID_SIZE_X] -= value;
      }
    }
  }
}

void Fluid2D::SolvePressure() {
  auto addv = [](const float *a, const float *b, float *res) {
    for (int i = 0; i < GRID_SIZE_X * GRID_SIZE_Y; i++) {
      res[i] = a[i] + b[i];
    }
  };

  auto mulv = [](const float *a, const float *b, float *res) {
    for (int i = 0; i < GRID_SIZE_X * GRID_SIZE_Y; i++) {
      res[i] = a[i] * b[i];
    }
  };
  auto mulf = [](const float *a, float s, float *res) {
    for (int i = 0; i < GRID_SIZE_X * GRID_SIZE_Y; i++) {
      res[i] = a[i] * s;
    }
  };
  auto subv = [](const float *a, const float *b, float *res) {
    for (int i = 0; i < GRID_SIZE_X * GRID_SIZE_Y; i++) {
      res[i] = a[i] - b[i];
    }
  };
  auto dot = [](const float *a, const float *b) {
    float ans = 0.0f;
    for (int i = 0; i < GRID_SIZE_X * GRID_SIZE_Y; i++) {
      ans += a[i] * b[i];
    }
    return ans;
  };
  auto len = [](const float *a) {
    float ans = 0.0f;
    for (int i = 0; i < GRID_SIZE_X * GRID_SIZE_Y; i++) {
      ans += a[i] * a[i];
    }
    return sqrt(ans);
  };
  SolvePressureImpactToDivergence(pressure, buffer);
  subv(divergence, buffer, r_vec);
  std::memcpy(p_vec, r_vec, sizeof(float) * GRID_SIZE_X * GRID_SIZE_Y);
  while (true) {
    SolvePressureImpactToDivergence(p_vec, Ap_vec);
    float rk2 = dot(r_vec, r_vec);
    float ak = rk2 / dot(p_vec, Ap_vec);
    mulf(p_vec, ak, buffer);
    addv(pressure, buffer, pressure);
    mulf(Ap_vec, ak, buffer);
    subv(r_vec, buffer, r_vec);
    auto rlen = len(r_vec);
    if (rlen < 1e-3f) {
      break;
    }
    float bk = dot(r_vec, r_vec) / rk2;
    mulf(p_vec, bk, buffer);
    addv(r_vec, buffer, p_vec);
  }
}

#include "fstream"

void Fluid2D::PrintAMatrix() {
  //  std::ofstream fout("matrix.csv");
  //  std::memset(pressure, 0, sizeof(pressure));
  //  fout << std::fixed << std::setprecision(7);
  //  for (int i = 0; i < GRID_SIZE_X * GRID_SIZE_Y; i++) {
  //    if (i)
  //      pressure[i - 1] = 0.0f;
  //    pressure[i] = 1.0f;
  //    SolvePressureImpactToDivergence(pressure, buffer);
  //    for (int j = 0; j < GRID_SIZE_X * GRID_SIZE_Y; j++) {
  //      fout << buffer[j] << ',';
  //    }
  //    fout << std::endl;
  //  }
  //  fout.close();
  //  exit(0);
}

void Fluid2D::PrintMatrixAndOpen(float *matrix) {
  std::ofstream file("matrix.csv");
  file << std::fixed << std::setprecision(8);
  for (int y = 0; y < GRID_SIZE_Y; y++) {
    for (int x = 0; x < GRID_SIZE_X; x++) {
      int id = y * GRID_SIZE_X + x;
      file << matrix[id] << ",";
    }
    file << std::endl;
  }
  file.close();
  if (std::system("start matrix.csv")) {
  }
}
