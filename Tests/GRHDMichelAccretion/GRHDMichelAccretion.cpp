/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#include "Coordinates.hpp"
#include "DimensionDefinitions.hpp"
#include "FixedBGMetric.hpp"
#include "IdealGasEOS.hpp"
#include "KerrSchild.hpp"
#include "StaticMetric.hpp"
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"
#include "UserVariables.hpp"
#include "Valencia.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace
{
struct MichelConfig
{
    double mass = 1.0;
    double gamma = 5.0 / 3.0;
    double sonic_radius = 8.0;
    double sonic_density = 1.0e-6;
    double r_min = 4.0;
    double r_max = 40.0;
    int num_points = 512;
    double root_tolerance = 1.0e-13;
};

struct MichelConstants
{
    double polytropic_K = 0.0;
    double mass_flux = 0.0;
    double bernoulli = 0.0;
    double sonic_ur = 0.0;
    double sonic_sound_speed_squared = 0.0;
};

struct MichelPoint
{
    double radius = 0.0;
    double rho = 0.0;
    double pressure = 0.0;
    double eps = 0.0;
    double ur = 0.0;
    double vr = 0.0;
    double lapse = 1.0;
    double radial_metric = 1.0;
    double sound_speed_squared = 0.0;
    double mass_flux = 0.0;
    double bernoulli = 0.0;
};

bool check_close(const std::string &name, double value, double expected,
                 double tolerance)
{
    const double scale = std::max(1.0, std::abs(expected));
    const double error = std::abs(value - expected) / scale;
    if (error > tolerance)
    {
        std::cout << name << " mismatch: got " << value << ", expected "
                  << expected << ", relative error " << error << std::endl;
        return false;
    }
    return true;
}

GRHD::CellGeometry geometry_at(const MichelConfig &config, double radius)
{
    KerrSchild::params_t bg_params;
    bg_params.mass = config.mass;
    bg_params.spin = 0.0;
    bg_params.center = {{0.0, 0.0, 0.0}};

    KerrSchild background(bg_params, 1.0);
    Coordinates<double> coords(IntVect(D_DECL(0, 0, 0)), 1.0,
                               {0.0, 0.0, 0.0});
    coords.x = radius;
    coords.y = 0.0;
    coords.z = 0.0;
    return GRHD::make_cell_geometry_from_background(background, coords);
}

double pressure_from_density(const MichelConstants &constants,
                             const MichelConfig &config, double rho)
{
    return constants.polytropic_K * std::pow(rho, config.gamma);
}

double specific_enthalpy(const MichelConstants &constants,
                         const MichelConfig &config, double rho)
{
    const double pressure = pressure_from_density(constants, config, rho);
    return 1.0 + config.gamma * pressure / ((config.gamma - 1.0) * rho);
}

MichelConstants make_constants(const MichelConfig &config)
{
    MichelConstants constants;
    constants.sonic_sound_speed_squared =
        config.mass / (2.0 * config.sonic_radius - 3.0 * config.mass);
    constants.sonic_ur =
        -std::sqrt(config.mass / (2.0 * config.sonic_radius));

    const double q = constants.sonic_sound_speed_squared /
                     (config.gamma - constants.sonic_sound_speed_squared *
                                         config.gamma / (config.gamma - 1.0));
    constants.polytropic_K =
        q / std::pow(config.sonic_density, config.gamma - 1.0);

    const double h_c = specific_enthalpy(constants, config,
                                         config.sonic_density);
    const double f_c =
        GRHD::schwarzschild_areal_f(config.mass, config.sonic_radius);
    constants.bernoulli =
        h_c * std::sqrt(f_c + constants.sonic_ur * constants.sonic_ur);
    constants.mass_flux = config.sonic_radius * config.sonic_radius *
                          config.sonic_density * constants.sonic_ur;
    return constants;
}

double bernoulli_residual(const MichelConstants &constants,
                          const MichelConfig &config, double radius,
                          double rho)
{
    const double f = GRHD::schwarzschild_areal_f(config.mass, radius);
    const double ur = constants.mass_flux / (radius * radius * rho);
    const double h = specific_enthalpy(constants, config, rho);
    return h * h * (f + ur * ur) -
           constants.bernoulli * constants.bernoulli;
}

double bisect_density_root(const MichelConstants &constants,
                           const MichelConfig &config, double radius,
                           double lower, double upper)
{
    double f_lower = bernoulli_residual(constants, config, radius, lower);
    double f_upper = bernoulli_residual(constants, config, radius, upper);
    if (f_lower == 0.0)
        return lower;
    if (f_upper == 0.0)
        return upper;
    if (f_lower * f_upper > 0.0)
        throw std::runtime_error("Michel density bracket has no sign change");

    for (int iter = 0; iter < 200; ++iter)
    {
        const double mid = 0.5 * (lower + upper);
        const double f_mid = bernoulli_residual(constants, config, radius, mid);
        if (std::abs(f_mid) < config.root_tolerance ||
            std::abs(upper - lower) <=
                config.root_tolerance * std::max(1.0e-300, std::abs(mid)))
            return mid;
        if (f_lower * f_mid <= 0.0)
        {
            upper = mid;
            f_upper = f_mid;
            (void)f_upper;
        }
        else
        {
            lower = mid;
            f_lower = f_mid;
        }
    }
    return 0.5 * (lower + upper);
}

double solve_density(const MichelConstants &constants,
                     const MichelConfig &config, double radius)
{
    if (std::abs(radius - config.sonic_radius) < 1.0e-12)
        return config.sonic_density;

    const double log_min = std::log(config.sonic_density * 1.0e-6);
    const double log_max = std::log(config.sonic_density * 1.0e6);
    const int samples = 6000;
    struct Bracket
    {
        double lower;
        double upper;
        double midpoint;
    };
    std::vector<Bracket> brackets;

    double last_rho = std::exp(log_min);
    double last_value = bernoulli_residual(constants, config, radius, last_rho);
    for (int i = 1; i <= samples; ++i)
    {
        const double weight = static_cast<double>(i) / samples;
        const double rho = std::exp(log_min + weight * (log_max - log_min));
        const double value = bernoulli_residual(constants, config, radius, rho);
        if (last_value == 0.0 || value == 0.0 || last_value * value < 0.0)
        {
            brackets.push_back({last_rho, rho, 0.5 * (last_rho + rho)});
        }
        last_rho = rho;
        last_value = value;
    }

    if (brackets.empty())
        throw std::runtime_error("Michel density solve found no root bracket");

    const bool inner_branch = radius < config.sonic_radius;
    const auto selected = std::find_if(
        brackets.begin(), brackets.end(), [&](const Bracket &bracket) {
            return inner_branch ? bracket.midpoint > config.sonic_density
                                : bracket.midpoint < config.sonic_density;
        });
    const Bracket &bracket =
        (selected != brackets.end()) ? *selected : brackets.front();
    return bisect_density_root(constants, config, radius, bracket.lower,
                               bracket.upper);
}

MichelPoint make_point(const MichelConstants &constants,
                       const MichelConfig &config, const IdealGasEOS &eos,
                       double radius)
{
    const double f = GRHD::schwarzschild_areal_f(config.mass, radius);
    if (f <= 0.0)
        throw std::runtime_error("Michel profile radius is inside the horizon");

    MichelPoint point;
    point.radius = radius;
    point.rho = solve_density(constants, config, radius);
    point.pressure = pressure_from_density(constants, config, point.rho);
    point.eps = point.pressure / ((config.gamma - 1.0) * point.rho);
    point.ur = constants.mass_flux / (radius * radius * point.rho);
    const auto geometry = geometry_at(config, radius);
    point.lapse = geometry.lapse;
    point.radial_metric = geometry.spatial_metric_LL[0][0];

    Tensor<1, double> shift_L;
    FOR(i) { shift_L[i] = 0.0; }
    FOR(i, j)
    {
        shift_L[i] += geometry.spatial_metric_LL[i][j] * geometry.shift_U[j];
    }

    const double shift_squared = TensorAlgebra::compute_dot_product(
        geometry.shift_U, geometry.shift_U, geometry.spatial_metric_LL);
    const double g_tt = -geometry.lapse * geometry.lapse + shift_squared;
    const double u_t = -std::sqrt(f + point.ur * point.ur);
    const double u_upper_t = (u_t - shift_L[0] * point.ur) / g_tt;
    const double W = geometry.lapse * u_upper_t;
    if (!(W > 1.0) || !std::isfinite(W))
        throw std::runtime_error("Michel Kerr-Schild Lorentz factor invalid");

    point.vr = point.ur / W + geometry.shift_U[0] / geometry.lapse;
    point.sound_speed_squared = eos.compute_sound_speed_squared(
        point.rho, point.eps, point.pressure);
    const double h = eos.compute_specific_enthalpy(point.rho, point.eps,
                                                   point.pressure);
    point.mass_flux = radius * radius * point.rho * point.ur;
    point.bernoulli = h * std::sqrt(f + point.ur * point.ur);
    return point;
}

GRHD::Primitive<double> primitive_from_point(const MichelPoint &point)
{
    GRHD::Primitive<double> primitive;
    primitive.rho = point.rho;
    primitive.eps = point.eps;
    primitive.pressure = point.pressure;
    FOR(i) { primitive.velocity_U[i] = 0.0; }
    primitive.velocity_U[0] = point.vr;
    return primitive;
}

std::string default_output_dir()
{
    struct stat info;
    if (stat("Tests/GRHDMichelAccretion", &info) == 0)
        return "Tests/GRHDMichelAccretion/output";
    return "output";
}

void ensure_output_dir(const std::string &dir)
{
    mkdir(dir.c_str(), 0755);
}

void write_profile(const std::vector<MichelPoint> &profile)
{
    const std::string out_dir = default_output_dir();
    ensure_output_dir(out_dir);
    const std::string path = out_dir + "/grhd_michel_profile.csv";
    std::ofstream out(path.c_str());
    if (!out)
        throw std::runtime_error("could not open " + path);
    out << "radius,rho,pressure,eps,ur,vr,lapse,grr,sound_speed_squared,"
        << "mass_flux,bernoulli\n";
    out << std::setprecision(17);
    for (const auto &point : profile)
    {
        out << point.radius << "," << point.rho << "," << point.pressure
            << "," << point.eps << "," << point.ur << "," << point.vr
            << "," << point.lapse << "," << point.radial_metric << ","
            << point.sound_speed_squared << "," << point.mass_flux << ","
            << point.bernoulli << "\n";
    }
}

bool validate_profile(const std::vector<MichelPoint> &profile,
                      const MichelConstants &constants,
                      const MichelConfig &config, const IdealGasEOS &eos)
{
    bool passed = true;
    double max_mass_flux_error = 0.0;
    double max_bernoulli_error = 0.0;
    double max_recovery_error = 0.0;
    double max_velocity_squared = 0.0;

    for (const auto &point : profile)
    {
        max_mass_flux_error = std::max(
            max_mass_flux_error,
            std::abs(point.mass_flux - constants.mass_flux) /
                std::max(std::abs(constants.mass_flux), 1.0e-300));
        max_bernoulli_error = std::max(
            max_bernoulli_error,
            std::abs(point.bernoulli - constants.bernoulli) /
                std::max(std::abs(constants.bernoulli), 1.0e-300));

        const auto geometry = geometry_at(config, point.radius);
        const auto &metric_LL = geometry.spatial_metric_LL;
        const auto &metric_UU = geometry.spatial_metric_UU;
        const auto primitive = primitive_from_point(point);
        const auto conserved = GRHD::compute_conserved(primitive, eos,
                                                       metric_LL);
        GRHD::RecoveryOptions recovery_options;
        recovery_options.max_iterations = 200;
        recovery_options.tolerance = 1.0e-14;
        recovery_options.pressure_floor = 1.0e-16;
        const auto recovered = GRHD::recover_primitive(
            conserved, eos, metric_UU, recovery_options);
        if (!recovered.success)
        {
            std::cout << "Michel primitive recovery failed at r = "
                      << point.radius << " with residual "
                      << recovered.residual << std::endl;
            passed = false;
            continue;
        }
        max_recovery_error = std::max(
            max_recovery_error,
            std::abs(recovered.primitive.rho - primitive.rho) /
                std::max(primitive.rho, 1.0e-300));
        max_recovery_error = std::max(
            max_recovery_error,
            std::abs(recovered.primitive.pressure - primitive.pressure) /
                std::max(primitive.pressure, 1.0e-300));
        max_recovery_error = std::max(
            max_recovery_error,
            std::abs(recovered.primitive.velocity_U[0] -
                     primitive.velocity_U[0]) /
                std::max(std::abs(primitive.velocity_U[0]), 1.0e-300));

        const double velocity_squared = GRHD::compute_velocity_squared(
            primitive.velocity_U, geometry.spatial_metric_LL);
        max_velocity_squared = std::max(max_velocity_squared, velocity_squared);
        if (!(point.rho > 0.0) || !(point.pressure > 0.0) ||
            velocity_squared >= 1.0)
        {
            std::cout << "Michel nonphysical state at r = " << point.radius
                      << std::endl;
            passed = false;
        }
    }

    const double sonic_sound_speed_expected =
        config.mass / (2.0 * config.sonic_radius - 3.0 * config.mass);
    passed &= check_close("Michel sonic sound speed squared",
                          constants.sonic_sound_speed_squared,
                          sonic_sound_speed_expected, 1.0e-14);
    passed &= check_close("Michel sonic ur squared",
                          constants.sonic_ur * constants.sonic_ur,
                          config.mass / (2.0 * config.sonic_radius),
                          1.0e-14);

    std::cout << "Michel diagnostics: max_mass_flux_rel_error = "
              << max_mass_flux_error
              << ", max_bernoulli_rel_error = " << max_bernoulli_error
              << ", max_recovery_rel_error = " << max_recovery_error
              << ", max_velocity_squared = " << max_velocity_squared
              << std::endl;

    passed &= max_mass_flux_error < 1.0e-9;
    passed &= max_bernoulli_error < 1.0e-9;
    passed &= max_recovery_error < 2.0e-6;
    passed &= max_velocity_squared < 1.0;
    return passed;
}
} // namespace

int main()
{
    try
    {
        const MichelConfig config;
        const IdealGasEOS eos(config.gamma);
        const MichelConstants constants = make_constants(config);
        std::vector<MichelPoint> profile;
        profile.reserve(config.num_points);
        for (int i = 0; i < config.num_points; ++i)
        {
            const double radius =
                config.r_min + (config.r_max - config.r_min) *
                                   (static_cast<double>(i) + 0.5) /
                                   static_cast<double>(config.num_points);
            profile.push_back(make_point(constants, config, eos, radius));
        }
        write_profile(profile);
        if (!validate_profile(profile, constants, config, eos))
            return 1;
        std::cout << "GRHD Michel accretion profile test passed..." << std::endl;
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cout << "GRHD Michel accretion profile test failed: "
                  << error.what() << std::endl;
        return 1;
    }
}
