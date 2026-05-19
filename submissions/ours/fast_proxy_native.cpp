#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

namespace {

inline int clamp_int(int v, int lo, int hi) {
  return std::max(lo, std::min(v, hi));
}

inline std::pair<int, int> grid_cell(double x, double y, double grid_w, double grid_h,
                                     int rows, int cols) {
  int row = static_cast<int>(std::floor(y / std::max(grid_h, 1e-12)));
  int col = static_cast<int>(std::floor(x / std::max(grid_w, 1e-12)));
  return {clamp_int(row, 0, rows - 1), clamp_int(col, 0, cols - 1)};
}

double abu_all_top(std::vector<double> values, double frac) {
  if (values.empty()) return 0.0;
  int cnt = static_cast<int>(std::floor(static_cast<double>(values.size()) * frac));
  if (cnt <= 0) return *std::max_element(values.begin(), values.end());
  cnt = std::min(cnt, static_cast<int>(values.size()));
  std::nth_element(values.begin(), values.end() - cnt, values.end());
  double sum = 0.0;
  for (auto it = values.end() - cnt; it != values.end(); ++it) sum += *it;
  return sum / static_cast<double>(cnt);
}

double abu_nonzero_top(const std::vector<double>& values, double frac, int total_count) {
  std::vector<double> nonzero;
  nonzero.reserve(values.size());
  for (double v : values) {
    if (v != 0.0) nonzero.push_back(v);
  }
  if (total_count < 10) {
    if (nonzero.empty()) return 0.0;
    return std::accumulate(nonzero.begin(), nonzero.end(), 0.0) /
           static_cast<double>(nonzero.size());
  }
  int cnt = static_cast<int>(std::floor(static_cast<double>(total_count) * frac));
  if (cnt <= 0) return values.empty() ? 0.0 : *std::max_element(values.begin(), values.end());
  if (nonzero.empty()) return 0.0;
  int k = std::min(cnt, static_cast<int>(nonzero.size()));
  std::nth_element(nonzero.begin(), nonzero.end() - k, nonzero.end());
  double sum = 0.0;
  for (auto it = nonzero.end() - k; it != nonzero.end(); ++it) sum += *it;
  return sum / static_cast<double>(cnt);
}

inline void owner_xy(int owner, const double* pos, int n_macros, const double* ports,
                     int n_ports, const double* offset, double& x, double& y) {
  if (owner < n_macros) {
    x = pos[2 * owner] + offset[0];
    y = pos[2 * owner + 1] + offset[1];
  } else {
    int port = owner - n_macros;
    if (port >= 0 && port < n_ports) {
      x = ports[2 * port];
      y = ports[2 * port + 1];
    } else {
      x = 0.0;
      y = 0.0;
    }
  }
}

void two_pin_route(std::vector<double>& h_route, std::vector<double>& v_route,
                   std::pair<int, int> source, std::pair<int, int> a,
                   std::pair<int, int> b, double weight, int cols) {
  std::pair<int, int> sink = (a == source) ? b : a;
  int row_min = std::min(sink.first, source.first);
  int row_max = std::max(sink.first, source.first);
  int col_min = std::min(sink.second, source.second);
  int col_max = std::max(sink.second, source.second);
  for (int col = col_min; col < col_max; ++col) {
    h_route[source.first * cols + col] += weight;
  }
  for (int row = row_min; row < row_max; ++row) {
    v_route[row * cols + sink.second] += weight;
  }
}

void l_route(std::vector<double>& h_route, std::vector<double>& v_route,
             std::vector<std::pair<int, int>> cells, double weight, int cols) {
  std::sort(cells.begin(), cells.end(), [](auto a, auto b) {
    if (a.second != b.second) return a.second < b.second;
    return a.first < b.first;
  });
  int y1 = cells[0].first, x1 = cells[0].second;
  int y2 = cells[1].first, x2 = cells[1].second;
  int y3 = cells[2].first, x3 = cells[2].second;
  for (int col = x1; col < x2; ++col) h_route[y1 * cols + col] += weight;
  for (int col = x2; col < x3; ++col) h_route[y2 * cols + col] += weight;
  for (int row = std::min(y1, y2); row < std::max(y1, y2); ++row) {
    v_route[row * cols + x2] += weight;
  }
  for (int row = std::min(y2, y3); row < std::max(y2, y3); ++row) {
    v_route[row * cols + x3] += weight;
  }
}

void t_route(std::vector<double>& h_route, std::vector<double>& v_route,
             std::vector<std::pair<int, int>> cells, double weight, int cols) {
  std::sort(cells.begin(), cells.end());
  int y1 = cells[0].first, x1 = cells[0].second;
  int y2 = cells[1].first, x2 = cells[1].second;
  int y3 = cells[2].first, x3 = cells[2].second;
  int xmin = std::min({x1, x2, x3});
  int xmax = std::max({x1, x2, x3});
  for (int col = xmin; col < xmax; ++col) h_route[y2 * cols + col] += weight;
  for (int row = std::min(y1, y2); row < std::max(y1, y2); ++row) {
    v_route[row * cols + x1] += weight;
  }
  for (int row = std::min(y2, y3); row < std::max(y2, y3); ++row) {
    v_route[row * cols + x3] += weight;
  }
}

void three_pin_route(std::vector<double>& h_route, std::vector<double>& v_route,
                     std::vector<std::pair<int, int>> cells, double weight, int cols) {
  std::sort(cells.begin(), cells.end(), [](auto a, auto b) {
    if (a.second != b.second) return a.second < b.second;
    return a.first < b.first;
  });
  int y1 = cells[0].first, x1 = cells[0].second;
  int y2 = cells[1].first, x2 = cells[1].second;
  int y3 = cells[2].first, x3 = cells[2].second;
  if (x1 < x2 && x2 < x3 && std::min(y1, y3) < y2 && std::max(y1, y3) > y2) {
    l_route(h_route, v_route, cells, weight, cols);
  } else if (x2 == x3 && x1 < x2 && y1 < std::min(y2, y3)) {
    for (int col = x1; col < x2; ++col) h_route[y1 * cols + col] += weight;
    for (int row = y1; row < std::max(y2, y3); ++row) v_route[row * cols + x2] += weight;
  } else if (y2 == y3) {
    for (int col = x1; col < x2; ++col) h_route[y1 * cols + col] += weight;
    for (int col = x2; col < x3; ++col) h_route[y2 * cols + col] += weight;
    for (int row = std::min(y2, y1); row < std::max(y2, y1); ++row) {
      v_route[row * cols + x2] += weight;
    }
  } else {
    t_route(h_route, v_route, cells, weight, cols);
  }
}

void macro_route(std::vector<double>& v_macro, std::vector<double>& h_macro,
                 const double* pos, const double* sizes, int idx, double grid_w,
                 double grid_h, int rows, int cols, double v_alloc, double h_alloc) {
  double x = pos[2 * idx];
  double y = pos[2 * idx + 1];
  double w = sizes[2 * idx];
  double h = sizes[2 * idx + 1];
  double x_min = x - 0.5 * w, x_max = x + 0.5 * w;
  double y_min = y - 0.5 * h, y_max = y + 0.5 * h;
  auto bl = grid_cell(x_min, y_min, grid_w, grid_h, rows, cols);
  auto ur = grid_cell(x_max, y_max, grid_w, grid_h, rows, cols);
  bool partial_v = false;
  bool partial_h = false;
  struct Rec { int row; int col; double xd; double yd; };
  std::vector<Rec> overlaps;
  for (int row = bl.first; row <= ur.first; ++row) {
    double gy0 = row * grid_h, gy1 = (row + 1) * grid_h;
    for (int col = bl.second; col <= ur.second; ++col) {
      double gx0 = col * grid_w, gx1 = (col + 1) * grid_w;
      double xd = std::max(0.0, std::min(x_max, gx1) - std::max(x_min, gx0));
      double yd = std::max(0.0, std::min(y_max, gy1) - std::max(y_min, gy0));
      if (xd <= 0.0 || yd <= 0.0) continue;
      if (ur.first != bl.first &&
          ((row == bl.first && std::abs(yd - grid_h) > 1e-5) ||
           (row == ur.first && std::abs(yd - grid_h) > 1e-5))) {
        partial_v = true;
      }
      if (ur.second != bl.second &&
          ((col == bl.second && std::abs(xd - grid_w) > 1e-5) ||
           (col == ur.second && std::abs(xd - grid_w) > 1e-5))) {
        partial_h = true;
      }
      int flat = row * cols + col;
      v_macro[flat] += xd * v_alloc;
      h_macro[flat] += yd * h_alloc;
      overlaps.push_back({row, col, xd, yd});
    }
  }
  if (partial_v) {
    for (const auto& rec : overlaps) {
      if (rec.row == ur.first) v_macro[rec.row * cols + rec.col] -= rec.xd * v_alloc;
    }
  }
  if (partial_h) {
    for (const auto& rec : overlaps) {
      if (rec.col == ur.second) h_macro[rec.row * cols + rec.col] -= rec.yd * h_alloc;
    }
  }
}

}  // namespace

extern "C" int score_proxy_native(
    int n_macros, int n_hard, int n_ports, int n_nets, int grid_rows, int grid_cols,
    double canvas_w, double canvas_h, double hroutes_per_micron, double vroutes_per_micron,
    int smooth_range, double h_alloc, double v_alloc, double net_count,
    const double* pos, const double* sizes, const double* ports,
    const int* net_starts, const int* owners, const double* offsets,
    const int* source_owner, const double* source_offsets,
    const double* hpwl_weights, const double* route_weights, double* out) {
  try {
    int grid_size = grid_rows * grid_cols;
    double grid_w = canvas_w / std::max(1, grid_cols);
    double grid_h = canvas_h / std::max(1, grid_rows);

    double total_hpwl = 0.0;
    for (int n = 0; n < n_nets; ++n) {
      int s = net_starts[n], e = net_starts[n + 1];
      if (s >= e) continue;
      double min_x = 1e100, min_y = 1e100, max_x = -1e100, max_y = -1e100;
      for (int p = s; p < e; ++p) {
        double x, y;
        owner_xy(owners[p], pos, n_macros, ports, n_ports, &offsets[2 * p], x, y);
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
      }
      total_hpwl += hpwl_weights[n] * ((max_x - min_x) + (max_y - min_y));
    }
    double wl_cost = total_hpwl / std::max((canvas_w + canvas_h) * net_count, 1e-12);

    std::vector<double> occupied(grid_size, 0.0);
    for (int i = 0; i < n_macros; ++i) {
      double x = pos[2 * i], y = pos[2 * i + 1];
      double w = sizes[2 * i], h = sizes[2 * i + 1];
      double x_min = x - 0.5 * w, x_max = x + 0.5 * w;
      double y_min = y - 0.5 * h, y_max = y + 0.5 * h;
      auto bl = grid_cell(x_min, y_min, grid_w, grid_h, grid_rows, grid_cols);
      auto ur = grid_cell(x_max, y_max, grid_w, grid_h, grid_rows, grid_cols);
      for (int row = bl.first; row <= ur.first; ++row) {
        double gy0 = row * grid_h, gy1 = (row + 1) * grid_h;
        double oy = std::min(y_max, gy1) - std::max(y_min, gy0);
        if (oy <= 0.0) continue;
        for (int col = bl.second; col <= ur.second; ++col) {
          double gx0 = col * grid_w, gx1 = (col + 1) * grid_w;
          double ox = std::min(x_max, gx1) - std::max(x_min, gx0);
          if (ox > 0.0) occupied[row * grid_cols + col] += ox * oy;
        }
      }
    }
    double grid_area = std::max(grid_w * grid_h, 1e-12);
    for (double& v : occupied) v /= grid_area;
    double density_cost = 0.5 * abu_nonzero_top(occupied, 0.10, grid_size);

    std::vector<double> h_route(grid_size, 0.0), v_route(grid_size, 0.0);
    std::vector<double> h_macro(grid_size, 0.0), v_macro(grid_size, 0.0);
    for (int n = 0; n < n_nets; ++n) {
      int s = net_starts[n], e = net_starts[n + 1];
      if (s >= e) continue;
      double sx, sy;
      owner_xy(source_owner[n], pos, n_macros, ports, n_ports, &source_offsets[2 * n], sx, sy);
      auto source = grid_cell(sx, sy, grid_w, grid_h, grid_rows, grid_cols);
      std::set<std::pair<int, int>> unique;
      unique.insert(source);
      for (int p = s; p < e; ++p) {
        double x, y;
        owner_xy(owners[p], pos, n_macros, ports, n_ports, &offsets[2 * p], x, y);
        unique.insert(grid_cell(x, y, grid_w, grid_h, grid_rows, grid_cols));
      }
      if (unique.size() == 2) {
        auto it = unique.begin();
        auto a = *it++;
        auto b = *it;
        two_pin_route(h_route, v_route, source, a, b, route_weights[n], grid_cols);
      } else if (unique.size() == 3) {
        std::vector<std::pair<int, int>> cells(unique.begin(), unique.end());
        three_pin_route(h_route, v_route, cells, route_weights[n], grid_cols);
      } else if (unique.size() > 3) {
        for (const auto& cell : unique) {
          if (cell != source) two_pin_route(h_route, v_route, source, source, cell, route_weights[n], grid_cols);
        }
      }
    }

    for (int i = 0; i < n_hard; ++i) {
      macro_route(v_macro, h_macro, pos, sizes, i, grid_w, grid_h, grid_rows, grid_cols, v_alloc, h_alloc);
    }

    double grid_v_routes = std::max(grid_w * vroutes_per_micron, 1e-12);
    double grid_h_routes = std::max(grid_h * hroutes_per_micron, 1e-12);
    for (int i = 0; i < grid_size; ++i) {
      v_route[i] /= grid_v_routes;
      h_route[i] /= grid_h_routes;
      v_macro[i] /= grid_v_routes;
      h_macro[i] /= grid_h_routes;
    }

    if (smooth_range > 0) {
      std::vector<double> v_tmp(grid_size, 0.0), h_tmp(grid_size, 0.0);
      for (int row = 0; row < grid_rows; ++row) {
        for (int col = 0; col < grid_cols; ++col) {
          int lp = std::max(0, col - smooth_range);
          int rp = std::min(grid_cols - 1, col + smooth_range);
          double val = v_route[row * grid_cols + col] / static_cast<double>(rp - lp + 1);
          for (int ptr = lp; ptr <= rp; ++ptr) v_tmp[row * grid_cols + ptr] += val;
        }
      }
      for (int row = 0; row < grid_rows; ++row) {
        for (int col = 0; col < grid_cols; ++col) {
          int lp = std::max(0, row - smooth_range);
          int up = std::min(grid_rows - 1, row + smooth_range);
          double val = h_route[row * grid_cols + col] / static_cast<double>(up - lp + 1);
          for (int ptr = lp; ptr <= up; ++ptr) h_tmp[ptr * grid_cols + col] += val;
        }
      }
      v_route.swap(v_tmp);
      h_route.swap(h_tmp);
    }

    std::vector<double> cong;
    cong.reserve(2 * grid_size);
    for (int i = 0; i < grid_size; ++i) cong.push_back(v_route[i] + v_macro[i]);
    for (int i = 0; i < grid_size; ++i) cong.push_back(h_route[i] + h_macro[i]);
    double congestion_cost = abu_all_top(std::move(cong), 0.05);

    int overlaps = 0;
    for (int i = 0; i < n_hard; ++i) {
      double xi = pos[2 * i], yi = pos[2 * i + 1];
      double wi = sizes[2 * i], hi = sizes[2 * i + 1];
      for (int j = i + 1; j < n_hard; ++j) {
        double xj = pos[2 * j], yj = pos[2 * j + 1];
        double wj = sizes[2 * j], hj = sizes[2 * j + 1];
        if (std::abs(xi - xj) < 0.5 * (wi + wj) && std::abs(yi - yj) < 0.5 * (hi + hj)) {
          overlaps += 1;
        }
      }
    }

    out[0] = wl_cost + 0.5 * density_cost + 0.5 * congestion_cost;
    out[1] = wl_cost;
    out[2] = density_cost;
    out[3] = congestion_cost;
    out[4] = static_cast<double>(overlaps);
    return 0;
  } catch (...) {
    return -1;
  }
}
