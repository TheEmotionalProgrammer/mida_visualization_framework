#include "gradient_volume.h"
#include <algorithm>
#include <exception>
#include <glm/geometric.hpp>
#include <glm/vector_relational.hpp>
#include <gsl/span>

namespace volume {

// Compute the maximum magnitude from all gradient voxels
static float computeMaxMagnitude(gsl::span<const GradientVoxel> data)
{
    return std::max_element(
        std::begin(data),
        std::end(data),
        [](const GradientVoxel& lhs, const GradientVoxel& rhs) {
            return lhs.magnitude < rhs.magnitude;
        })
        ->magnitude;
}

// Compute the minimum magnitude from all gradient voxels
static float computeMinMagnitude(gsl::span<const GradientVoxel> data)
{
    return std::min_element(
        std::begin(data),
        std::end(data),
        [](const GradientVoxel& lhs, const GradientVoxel& rhs) {
            return lhs.magnitude < rhs.magnitude;
        })
        ->magnitude;
}

// Compute a gradient volume from a volume
static std::vector<GradientVoxel> computeGradientVolume(const Volume& volume)
{
    const auto dim = volume.dims();

    std::vector<GradientVoxel> out(static_cast<size_t>(dim.x * dim.y * dim.z));
    for (int z = 1; z < dim.z - 1; z++) {
        for (int y = 1; y < dim.y - 1; y++) {
            for (int x = 1; x < dim.x - 1; x++) {
                const float gx = (volume.getVoxel(x + 1, y, z) - volume.getVoxel(x - 1, y, z)) / 2.0f;
                const float gy = (volume.getVoxel(x, y + 1, z) - volume.getVoxel(x, y - 1, z)) / 2.0f;
                const float gz = (volume.getVoxel(x, y, z + 1) - volume.getVoxel(x, y, z - 1)) / 2.0f;

                const glm::vec3 v { gx, gy, gz };
                const size_t index = static_cast<size_t>(x + dim.x * (y + dim.y * z));
                out[index] = GradientVoxel { v, glm::length(v) };
            }
        }
    }
    return out;
}

GradientVolume::GradientVolume(const Volume& volume)
    : m_dim(volume.dims())
    , m_data(computeGradientVolume(volume))
    , m_minMagnitude(computeMinMagnitude(m_data))
    , m_maxMagnitude(computeMaxMagnitude(m_data))
{
}

float GradientVolume::maxMagnitude() const
{
    return m_maxMagnitude;
}

float GradientVolume::minMagnitude() const
{
    return m_minMagnitude;
}

glm::ivec3 GradientVolume::dims() const
{
    return m_dim;
}

// This function returns a gradientVoxel at coord based on the current interpolation mode.
GradientVoxel GradientVolume::getGradientInterpolate(const glm::vec3& coord) const
{
    switch (interpolationMode) {
    case InterpolationMode::NearestNeighbour: {
        return getGradientNearestNeighbor(coord);
    }
    case InterpolationMode::Linear: {
        return getGradientLinearInterpolate(coord);
    }
    case InterpolationMode::Cubic: {
        // No cubic in this case, linear is good enough for the gradient.
        return getGradientLinearInterpolate(coord);
    }
    default: {
        throw std::exception();
    }
    };
}

// This function returns the nearest neighbour given a position in the volume given by coord.
// Notice that in this framework we assume that the distance between neighbouring voxels is 1 in all directions
GradientVoxel GradientVolume::getGradientNearestNeighbor(const glm::vec3& coord) const
{
    if (glm::any(glm::lessThan(coord, glm::vec3(0))) || glm::any(glm::greaterThanEqual(coord, glm::vec3(m_dim))))
        return { glm::vec3(0.0f), 0.0f };

    auto roundToPositiveInt = [](float f) {
        return static_cast<int>(f + 0.5f);
    };

    return getGradient(roundToPositiveInt(coord.x), roundToPositiveInt(coord.y), roundToPositiveInt(coord.z));
}

// ======= TODO : IMPLEMENT ========
// Returns the trilinearly interpolated gradinet at the given coordinate.
// Use the linearInterpolate function that you implemented below.
GradientVoxel GradientVolume::getGradientLinearInterpolate(const glm::vec3& coord) const
{   

    if (glm::any(glm::lessThan(coord, glm::vec3(0))) || glm::any(glm::greaterThanEqual(coord + 1.0f, glm::vec3(m_dim))))
        return { glm::vec3(0.0f), 0.0f };

    //get the 8 points around the coord
    int x0 = static_cast<int>(floor(coord.x));
    int x1 = static_cast<int>(ceil(coord.x));
    int y0 = static_cast<int>(floor(coord.y));
    int y1 = static_cast<int>(ceil(coord.y));
    int z0 = static_cast<int>(floor(coord.z));
    int z1 = static_cast<int>(ceil(coord.z));

    //get the 8 gradients
    GradientVoxel g000 = getGradient(x0, y0, z0);
    GradientVoxel g001 = getGradient(x0, y0, z1);
    GradientVoxel g010 = getGradient(x0, y1, z0);
    GradientVoxel g011 = getGradient(x0, y1, z1);
    GradientVoxel g100 = getGradient(x1, y0, z0);
    GradientVoxel g101 = getGradient(x1, y0, z1);
    GradientVoxel g110 = getGradient(x1, y1, z0);
    GradientVoxel g111 = getGradient(x1, y1, z1);
   // Calculate the interpolation factors for each axis
    float fx = coord.x - static_cast<float>(x0);
    float fy = coord.y - static_cast<float>(y0);
    float fz = coord.z - static_cast<float>(z0);

    // Interpolate along x for the bottom and top of the voxel
    GradientVoxel g00 = linearInterpolate(g000, g100, fx);
    GradientVoxel g01 = linearInterpolate(g001, g101, fx);
    GradientVoxel g10 = linearInterpolate(g010, g110, fx);
    GradientVoxel g11 = linearInterpolate(g011, g111, fx);

    // Interpolate along y
    GradientVoxel g0 = linearInterpolate(g00, g10, fy);
    GradientVoxel g1 = linearInterpolate(g01, g11, fy);

    // Interpolate along z
    GradientVoxel g = linearInterpolate(g0, g1, fz);

    return g;
}

// ======= TODO : IMPLEMENT ========
// This function should linearly interpolates the value from g0 to g1 given the factor (t).
// At t=0, linearInterpolate should return g0 and at t=1 it returns g1.
GradientVoxel GradientVolume::linearInterpolate(const GradientVoxel& g0, const GradientVoxel& g1, float factor)
{   
    factor = glm::clamp(factor, 0.0f, 1.0f);
    //linear interpolation of the magnitude
    float magnitude = g0.magnitude + factor * (g1.magnitude - g0.magnitude);

    //linear interpolation of the direction
    glm::vec3 direction = g0.dir + factor * (g1.dir - g0.dir);

    return GradientVoxel { direction, magnitude };
}

// This function returns a gradientVoxel without using interpolation
GradientVoxel GradientVolume::getGradient(int x, int y, int z) const
{
    const size_t i = static_cast<size_t>(x + m_dim.x * (y + m_dim.y * z));
    return m_data[i];
}
}