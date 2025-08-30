#include <moc/moc_plotter.hpp>
#include <moc/cartesian_2d.hpp>
#include <utils/logging.hpp>
#include <utils/scarabee_exception.hpp>

#include <sstream>
#include <random>

namespace scarabee {

MOCPlotter::MOCPlotter(const MOCDriver* moc)
    : moc_(moc),
      geom_(nullptr),
      cell_id_to_color(),
      material_id_to_color(),
      material_name_to_color(),
      image(500, 500),
      create_color_mutex(),
      height(10.),
      width(10.),
      ox(0.),
      oy(0.),
      mx(0.),
      my(0.),
      mufsr{nullptr, 0},
      mxs(nullptr),
      background(),
      colorby(ColorBy::MaterialName),
      must_rerender(true),
      outline_boundaries(true) {
  if (moc_ == nullptr) {
    std::stringstream mssg;
    mssg << "Provided MOCDriver is a nullptr.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  geom_ = moc_->geometry().get();
}

void MOCPlotter::render() {
  // We put a Dockspace over the entire viewport.
  ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

  this->render_viewport();
  this->render_controls();
}

void MOCPlotter::render_viewport() {
  // Get IO instance, as we will need it for certain things
  const auto& io = ImGui::GetIO();

  // Make window
  ImGui::SetNextWindowSize({500, 500}, ImGuiCond_Once);
  ImGui::Begin("Viewport");

  // First, get the window size. If the size doesn't match the current
  // image size, we must set must_rerender to true.
  ImVec2 size = ImGui::GetContentRegionAvail();
  std::uint32_t wwidth = static_cast<std::uint32_t>(size[0]);
  std::uint32_t wheight = static_cast<std::uint32_t>(size[1]);
  if ((wwidth != image.width()) || (wheight != image.height())) {
    image.resize(wheight, wwidth);

    dist_per_pixel = width / static_cast<double>(image.width());
    height = static_cast<double>(image.height()) * dist_per_pixel;

    must_rerender = true;
  }

  // Now we check to see if the user is dragging their mouse, to
  // change the origin of the plot
  if (ImGui::IsWindowHovered() && ImGui::IsMouseDragging(0)) {
    ImVec2 mouse_drag = io.MouseDelta;
    if (mouse_drag[0] != 0. || mouse_drag[1] != 0.) {
      ox -= dist_per_pixel * mouse_drag[0];
      oy += dist_per_pixel * mouse_drag[1];
      must_rerender = true;
    }
  }

  // Now we check to see if the user is scrolling, which can
  // resize the image zoom, by chaning the width and height.
  if (ImGui::IsWindowHovered() && std::abs(io.MouseWheel) > 0.1) {
    // Change width
    width += 0.05 * io.MouseWheel * width;
    if (width < 1.E-6) width = 1.E-6;

    // Must recalculate height
    dist_per_pixel = width / static_cast<double>(image.width());
    height = static_cast<double>(image.height()) * dist_per_pixel;

    must_rerender = true;
  }

  // If we window must be rerendered, we do that now
  if (must_rerender) {
    this->render_image();
    must_rerender = false;

    // Now we need to send the image to the GPU.
    image.send_to_gpu();
  }

  // Get upper-left corner of image, in Window frame (pixels)
  const ImVec2 img_pos = ImGui::GetCursorPos();

  // Now we need to add the image to the window
  void* texture_id = reinterpret_cast<void*>(
      static_cast<intptr_t>(image.ogl_texture_id().value()));
  ImGui::Image(texture_id, ImVec2(static_cast<float>(image.width()),
                                  static_cast<float>(image.height())));

  // Now we get the mouse position, so the user can identify cells
  // and materials
  if (ImGui::IsWindowHovered() && !ImGui::IsMouseDragging(0)) {
    // Get the new mouse coordinates in screen space (pixels)
    const ImVec2 mouse_pos = ImGui::GetMousePos();

    // Get window position in screen space (pixels)
    const ImVec2 window_pos = ImGui::GetWindowPos();

    // Get image in screen space (pixel)
    ImVec2 img_pos_on_screen;
    img_pos_on_screen[0] = img_pos[0] + window_pos[0];
    img_pos_on_screen[1] = img_pos[1] + window_pos[1];

    // Get the position of mouse relative to image (pixel)
    ImVec2 mouse_img_pos;
    mouse_img_pos[0] = mouse_pos[0] - img_pos_on_screen[0];
    mouse_img_pos[1] = mouse_pos[1] - img_pos_on_screen[1];

    mouse_img_pos[0] -= 0.5f * static_cast<float>(image.width());
    mouse_img_pos[1] -= 0.5f * static_cast<float>(image.height());

    // Convert the image position to physical position
    mx = ox + dist_per_pixel * mouse_img_pos[0];
    my = oy - dist_per_pixel * mouse_img_pos[1];

    // Initialize a tracker with mouse position
    Vector mp(mx, my);
    Direction mu(1., 0.);

    mufsr = geom_->get_fsr(mp, mu);

    if (mufsr.fsr) {
      mxs = mufsr.fsr->xs().get();
      mfsri = moc_->get_fsr_indx(mufsr);
    } else {
      mxs = nullptr;
    }
  }

  // Capture a right-click to bring up color changer
  if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(1) && mufsr.fsr)
    ImGui::OpenPopup("Select Color");
  if (ImGui::BeginPopup("Select Color")) {
    ImApp::Pixel color = get_color(mufsr);
    ImVec4 fcolor(static_cast<float>(color.r()) / 255.f,
                  static_cast<float>(color.g()) / 255.f,
                  static_cast<float>(color.b()) / 255.f,
                  static_cast<float>(color.a()) / 255.f);

    if (colorby == ColorBy::Cell) {
      ImGui::Text("FSR ID: %li", mufsr.fsr->id());
    } else {
      ImGui::Text("Material Name: %s", mxs->name().data());
    }

    if (ImGui::ColorEdit3("", reinterpret_cast<float*>(&fcolor))) {
      color.r() = static_cast<uint8_t>(fcolor.x * 255.f);
      color.g() = static_cast<uint8_t>(fcolor.y * 255.f);
      color.b() = static_cast<uint8_t>(fcolor.z * 255.f);

      if (colorby == ColorBy::Material)
        material_id_to_color[mxs] = color;
      else if (colorby == ColorBy::MaterialName)
        material_name_to_color[mxs->name()] = color;
      else
        cell_id_to_color[mufsr.fsr->id()] = color;

      must_rerender = true;
    }
    ImGui::EndPopup();
  }

  ImGui::End();
}

void MOCPlotter::render_controls() {
  ImGui::SetNextWindowSize({500, 500}, ImGuiCond_Once);
  ImGui::Begin("Controls");

  // Origin
  ImGui::Separator();
  ImGui::Text("Plot Origin");
  if (ImGui::InputDouble("X [cm]", &ox, 0., 0.)) must_rerender = true;
  if (ImGui::InputDouble("Y [cm]", &oy, 0., 0.)) must_rerender = true;

  // Physical Dimensions of plot
  ImGui::Separator();
  ImGui::Text("Width/Height");
  ImGui::RadioButton("Width", &adjust_w_or_h, 0);
  ImGui::SameLine();
  ImGui::RadioButton("Height", &adjust_w_or_h, 1);
  if (adjust_w_or_h == 0) {
    if (ImGui::InputDouble("Width [cm]", &width, 0., 0.)) {
      must_rerender = true;

      if (width < 1.E-6) width = 1.E-6;

      // Must recalculate height
      dist_per_pixel = width / static_cast<double>(image.width());
      height = static_cast<double>(image.height()) * dist_per_pixel;
    }
  } else if (adjust_w_or_h == 1) {
    if (ImGui::InputDouble("Height [cm]", &height, 0., 0.)) {
      must_rerender = true;

      if (height < 1.E-6) height = 1.E-6;

      // Must recalculate width
      dist_per_pixel = height / static_cast<double>(image.height());
      width = static_cast<double>(image.width()) * dist_per_pixel;
    }
  }
  ImGui::Text("Width [cm]: %f, Height [cm]: %f", width, height);

  // Color Method
  ImGui::Separator();
  ImGui::Text("Coloring");
  if (ImGui::RadioButton("Cell", reinterpret_cast<int*>(&colorby),
                         ColorBy::Cell))
    must_rerender = true;
  ImGui::SameLine();
  if (ImGui::RadioButton("Unique Cell", reinterpret_cast<int*>(&colorby),
                         ColorBy::UniqueCell))
    must_rerender = true;

  if (ImGui::RadioButton("Material", reinterpret_cast<int*>(&colorby),
                         ColorBy::Material))
    must_rerender = true;
  ImGui::SameLine();
  if (ImGui::RadioButton("Material Name", reinterpret_cast<int*>(&colorby),
                         ColorBy::MaterialName))
    must_rerender = true;

  if (ImGui::Checkbox("Mark Boundaries", &outline_boundaries))
    must_rerender = true;

  // Mouse Position
  ImGui::Separator();
  ImGui::Text("Mouse Position: (%f, %f)", mx, my);
  ImApp::Pixel color = get_color(mufsr);
  ImVec4 fcolor(static_cast<float>(color.r()) / 255.f,
                static_cast<float>(color.g()) / 255.f,
                static_cast<float>(color.b()) / 255.f,
                static_cast<float>(color.a()) / 255.f);

  if (!mufsr.fsr) {
    ImGui::Text("Cell for given position is not defined.");
  } else {
    ImGui::Text("FSR ID: %li", mufsr.fsr->id());
    ImGui::Text("FSR Instance: %li", mufsr.instance);
    ImGui::Text("FSR Index: %li", mfsri);
    ImGui::Text("FSR Area: %f", mufsr.fsr->volume());

    if (colorby == ColorBy::Cell &&
        ImGui::ColorEdit3("", reinterpret_cast<float*>(&fcolor))) {
      color.r() = static_cast<uint8_t>(fcolor.x * 255.f);
      color.g() = static_cast<uint8_t>(fcolor.y * 255.f);
      color.b() = static_cast<uint8_t>(fcolor.z * 255.f);

      cell_id_to_color[mufsr.fsr->id()] = color;

      must_rerender = true;
    }

    if (colorby == ColorBy::UniqueCell &&
        ImGui::ColorEdit3("", reinterpret_cast<float*>(&fcolor))) {
      color.r() = static_cast<uint8_t>(fcolor.x * 255.f);
      color.g() = static_cast<uint8_t>(fcolor.y * 255.f);
      color.b() = static_cast<uint8_t>(fcolor.z * 255.f);

      unique_cell_to_color[mufsr] = color;

      must_rerender = true;
    }
  }
  if (!mxs) {
    ImGui::Text("Material for given position is not defined.");
  } else {
    ImGui::Text("Material Name: %s", mxs->name().data());

    if (colorby == ColorBy::Material &&
        ImGui::ColorEdit3("", reinterpret_cast<float*>(&fcolor))) {
      color.r() = static_cast<uint8_t>(fcolor.x * 255.f);
      color.g() = static_cast<uint8_t>(fcolor.y * 255.f);
      color.b() = static_cast<uint8_t>(fcolor.z * 255.f);

      material_id_to_color[mxs] = color;

      must_rerender = true;
    }

    if (colorby == ColorBy::MaterialName &&
        ImGui::ColorEdit3("", reinterpret_cast<float*>(&fcolor))) {
      color.r() = static_cast<uint8_t>(fcolor.x * 255.f);
      color.g() = static_cast<uint8_t>(fcolor.y * 255.f);
      color.b() = static_cast<uint8_t>(fcolor.z * 255.f);

      material_name_to_color[mxs->name()] = color;

      must_rerender = true;
    }
  }

  // Save Image
  ImGui::Separator();
  if (ImGui::Button("Save Plot")) ImGui::OpenPopup("Save Plot");
  if (ImGui::BeginPopup("Save Plot")) {
    static char fn_str[500] = "";

    ImGui::InputTextWithHint("", "Enter File Name (i.e. reactor)", fn_str,
                             IM_ARRAYSIZE(fn_str));

    if (ImGui::Button("Save JPG")) {
      std::filesystem::path fname(fn_str);
      fname += ".jpg";
      image.save_jpg(fname);
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Save PNG")) {
      std::filesystem::path fname(fn_str);
      fname += ".png";
      image.save_png(fname);
      ImGui::CloseCurrentPopup();
    }

    ImGui::EndPopup();
  }

  ImGui::End();
}

void MOCPlotter::render_image() {
  {
    // Get the tracking direction
    const Direction u = this->get_tracking_direction();

// Go through each pixel
#ifdef ABEILLE_USE_OMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (uint32_t j = 0; j < image.height(); j++) {
      Vector strt = get_start_position(j);

      auto ufsr_r = geom_->get_fsr_r_local(strt, u);
      ImApp::Pixel pixel_color = this->get_color(ufsr_r.first);

      uint32_t i = 0;
      while (i < image.width()) {
        // Get the boundary distance
        double bound_distance = INF;
        if (ufsr_r.first.fsr) {
          bound_distance = ufsr_r.first.fsr->distance(ufsr_r.second, u);
        } else {
          bound_distance = geom_->distance(ufsr_r.second, u);
        }

        // Get the number of pixels till the boundary
        const double pixels_to_bound = bound_distance / dist_per_pixel;
        uint32_t npixels = static_cast<uint32_t>(std::round(pixels_to_bound));
        if (npixels > (image.width() - i) || bound_distance == INF) {
          npixels = image.width() - i;
        }
        const double npixels_dist =
            static_cast<double>(npixels) * dist_per_pixel;

        // Set all pixels
        for (uint32_t p = 0; p < npixels; p++) {
          if (i >= image.width()) break;
          if (outline_boundaries && npixels > 0 &&
              (p == npixels - 1 || i == image.width() - 1)) {
            image.at(j, i) = ImApp::Pixel(0, 0, 0);
          } else {
            image.at(j, i) = pixel_color;
          }
          i++;
        }
        if (i >= image.width()) break;

        // Cross boundary, update cell, and get new pixel
        if (pixels_to_bound - static_cast<double>(npixels) < 0.5) {
          strt = strt + (npixels_dist + dist_per_pixel) * u;
        } else {
          strt = strt + npixels_dist * u;
        }
        ufsr_r = geom_->get_fsr_r_local(strt, u);
        pixel_color = this->get_color(ufsr_r.first);
        if (pixels_to_bound - static_cast<double>(npixels) < 0.5) {
          if (i >= image.width()) break;
          image.at(j, i) = pixel_color;
          i++;
        }
      }  // while i < plot_width_
    }  // For j which is parallel
  }

  if (outline_boundaries) {
    // Get the tracking direction
    const Direction u = this->get_comp_tracking_direction();

    // Go through each pixel
#ifdef ABEILLE_USE_OMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (uint32_t i = 0; i < image.width(); i++) {
      Vector strt = get_comp_start_position(i);

      auto ufsr_r = geom_->get_fsr_r_local(strt, u);

      uint32_t j = 0;
      while (j < image.height()) {
        // Get the boundary distance
        double bound_distance = INF;
        if (ufsr_r.first.fsr) {
          bound_distance = ufsr_r.first.fsr->distance(ufsr_r.second, u);
        } else {
          bound_distance = geom_->distance(ufsr_r.second, u);
        }

        // Get the number of pixels till the boundary
        const double pixels_to_bound = bound_distance / dist_per_pixel;
        uint32_t npixels = static_cast<uint32_t>(std::round(pixels_to_bound));
        if (npixels > (image.height() - j) || bound_distance == INF) {
          npixels = image.height() - j;
        }
        const double npixels_dist =
            static_cast<double>(npixels) * dist_per_pixel;

        // Set all pixels
        for (uint32_t p = 0; p < npixels; p++) {
          if (j >= image.height()) break;
          if (npixels > 0 && (p == npixels - 1 || j == image.height() - 1)) {
            image.at(j, i) = ImApp::Pixel(0, 0, 0);
          }
          j++;
        }
        if (j >= image.height()) break;

        // Cross boundary, update cell, and get new pixel
        if (pixels_to_bound - static_cast<double>(npixels) < 0.5) {
          strt = strt + (npixels_dist + dist_per_pixel) * u;
        } else {
          strt = strt + npixels_dist * u;
        }
        ufsr_r = geom_->get_fsr_r_local(strt, u);
        if (pixels_to_bound - static_cast<double>(npixels) < 0.5) {
          if (j >= image.height()) break;
          j++;
        }
      }  // while j < image.height
    }  // For i which is parallel
  }
}

ImApp::Pixel MOCPlotter::get_random_color() {
  static std::minstd_rand rng_engn;
  static std::uniform_real_distribution unit_dist(0., 1.);

  uint8_t r = static_cast<uint8_t>(255.0 * unit_dist(rng_engn));
  uint8_t g = static_cast<uint8_t>(255.0 * unit_dist(rng_engn));
  uint8_t b = static_cast<uint8_t>(255.0 * unit_dist(rng_engn));

  return ImApp::Pixel(r, g, b);
}

ImApp::Pixel MOCPlotter::get_color(UniqueFSR ufsr) {
  ImApp::Pixel pixel_color = background;
  // If pointer isn't nullpntr, get pixel
  if (ufsr.fsr != nullptr) {
    if (colorby == ColorBy::Cell) {
      // Do same check twice with mutex to make thread safe
      auto it = cell_id_to_color.find(ufsr.fsr->id());
      if (it == cell_id_to_color.end()) {
        // Check if cell id is in id_to_pixel
        create_color_mutex.lock();
        if (cell_id_to_color.find(ufsr.fsr->id()) == cell_id_to_color.end()) {
          // Get new random color for id
          pixel_color = get_random_color();
          cell_id_to_color[ufsr.fsr->id()] = pixel_color;
        }
        create_color_mutex.unlock();
      } else {
        pixel_color = it->second;
      }
    } else if (colorby == ColorBy::UniqueCell) {
      // Do same check twice with mutex to make thread safe
      auto it = unique_cell_to_color.find(ufsr);
      if (it == unique_cell_to_color.end()) {
        // Check if cell id is in id_to_pixel
        create_color_mutex.lock();
        if (unique_cell_to_color.find(ufsr) == unique_cell_to_color.end()) {
          // Get new random color for id
          pixel_color = get_random_color();
          unique_cell_to_color[ufsr] = pixel_color;
        }
        create_color_mutex.unlock();
      } else {
        pixel_color = it->second;
      }
    } else if (colorby == ColorBy::Material) {
      // Color by material
      CrossSection* xs = ufsr.fsr->xs().get();
      // Do same check twice with mutex to make thread safe
      auto it = material_id_to_color.find(xs);
      if (it == material_id_to_color.end()) {
        // Check if cell id is in id_to_pixel
        create_color_mutex.lock();
        if (material_id_to_color.find(xs) == material_id_to_color.end()) {
          // Get new random color for id
          pixel_color = get_random_color();
          material_id_to_color[xs] = pixel_color;
        }
        create_color_mutex.unlock();
      } else {
        pixel_color = it->second;
      }
    } else {
      // Color by material name
      CrossSection* xs = ufsr.fsr->xs().get();
      // Do same check twice with mutex to make thread safe
      auto it = material_name_to_color.find(xs->name());
      if (it == material_name_to_color.end()) {
        // Check if cell id is in id_to_pixel
        create_color_mutex.lock();
        if (material_name_to_color.find(xs->name()) ==
            material_name_to_color.end()) {
          // Get new random color for id
          pixel_color = get_random_color();
          material_name_to_color[xs->name()] = pixel_color;
        }
        create_color_mutex.unlock();
      } else {
        pixel_color = it->second;
      }
    }
  }
  return pixel_color;
}

Direction MOCPlotter::get_tracking_direction() const {
  return Direction(1., 0.);
}

Direction MOCPlotter::get_comp_tracking_direction() const {
  return Direction(0., -1.);
}

Vector MOCPlotter::get_start_position(uint64_t j) const {
  // Make sure indicies are valid. j goes down so is height, i goes
  // across so is width
  if (j >= image.height()) {
    std::stringstream mssg;
    mssg << "Trying to deffine invalid pixel for plot.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  // x is i going across, j is y going up
  // First get height and width of a pixel
  const double dy = height / static_cast<double>(image.height());

  // Get upper corner off plot
  const double x_low = ox - (0.5 * width);
  const double y_high = oy + (0.5 * height);

  // Get coordinate of pixel
  const double x = x_low;
  const double y = y_high - (static_cast<double>(j) + 0.5) * dy;

  // Return coordinate
  return {x, y};
}

Vector MOCPlotter::get_comp_start_position(uint64_t i) const {
  // Make sure indicies are valid. i goes across so is width, j goes
  // down so is height
  if (i >= image.width()) {
    std::stringstream mssg;
    mssg << "Trying to deffine invalid pixel for plot.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  // x is i going across, j is y going down
  // First get height and width of a pixel
  const double dx = width / static_cast<double>(image.width());

  // Get upper corner off plot
  const double x_low = ox - (0.5 * width);
  const double y_high = oy + (0.5 * height);

  // Get coordinate of pixel
  const double x = (static_cast<double>(i) + 0.5) * dx + x_low;
  const double y = y_high;

  // Return coordinate
  return {x, y};
}

}  // namespace scarabee
