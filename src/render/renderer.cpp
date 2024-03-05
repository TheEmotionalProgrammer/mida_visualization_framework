#include "renderer.h"
#include <algorithm>
#include <algorithm> // std::fill
#include <cmath>
#include <functional>
#include <glm/common.hpp>
#include <glm/gtx/component_wise.hpp>
#include <iostream>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tuple>

namespace render {

// The renderer is passed a pointer to the volume, gradinet volume, camera and an initial renderConfig.
// The camera being pointed to may change each frame (when the user interacts). When the renderConfig
// changes the setConfig function is called with the updated render config. This gives the Renderer an
// opportunity to resize the framebuffer.
Renderer::Renderer(
    const volume::Volume* pVolume,
    const volume::GradientVolume* pGradientVolume,
    const render::RayTraceCamera* pCamera,
    const RenderConfig& initialConfig)
    : m_pVolume(pVolume)
    , m_pGradientVolume(pGradientVolume)
    , m_pCamera(pCamera)
    , m_config(initialConfig)
{
    resizeImage(initialConfig.renderResolution);
}

// Set a new render config if the user changed the settings.
void Renderer::setConfig(const RenderConfig& config)
{
    if (config.renderResolution != m_config.renderResolution)
        resizeImage(config.renderResolution);

    m_config = config;
}

// Resize the framebuffer and fill it with black pixels.
void Renderer::resizeImage(const glm::ivec2& resolution)
{
    m_frameBuffer.resize(size_t(resolution.x) * size_t(resolution.y), glm::vec4(0.0f));
}

// Clear the framebuffer by setting all pixels to black.
void Renderer::resetImage()
{
    std::fill(std::begin(m_frameBuffer), std::end(m_frameBuffer), glm::vec4(0.0f));
}

// Return a VIEW into the framebuffer. This view is merely a reference to the m_frameBuffer member variable.
// This does NOT make a copy of the framebuffer.
gsl::span<const glm::vec4> Renderer::frameBuffer() const
{
    return m_frameBuffer;
}

// Main render function. It computes an image according to the current renderMode.
// Multithreading is enabled in Release/RelWithDebInfo modes. In Debug mode multithreading is disabled to make debugging easier.
void Renderer::render()
{
    resetImage();

    static constexpr float sampleStep = 1.0f;
    const glm::vec3 planeNormal = -glm::normalize(m_pCamera->forward());
    const glm::vec3 volumeCenter = glm::vec3(m_pVolume->dims()) / 2.0f;
    const Bounds bounds { glm::vec3(0.0f), glm::vec3(m_pVolume->dims() - glm::ivec3(1)) };

    // 0 = sequential (single-core), 1 = TBB (multi-core)
#ifdef NDEBUG
    // If NOT in debug mode then enable parallelism using the TBB library (Intel Threaded Building Blocks).
#define PARALLELISM 1
#else
    // Disable multi threading in debug mode.
#define PARALLELISM 0
#endif

#if PARALLELISM == 0
    // Regular (single threaded) for loops.
    for (int x = 0; x < m_config.renderResolution.x; x++) {
        for (int y = 0; y < m_config.renderResolution.y; y++) {
#else
    // Parallel for loop (in 2 dimensions) that subdivides the screen into tiles.
    const tbb::blocked_range2d<int> screenRange { 0, m_config.renderResolution.y, 0, m_config.renderResolution.x };
        tbb::parallel_for(screenRange, [&](tbb::blocked_range2d<int> localRange) {
        // Loop over the pixels in a tile. This function is called on multiple threads at the same time.
        for (int y = std::begin(localRange.rows()); y != std::end(localRange.rows()); y++) {
            for (int x = std::begin(localRange.cols()); x != std::end(localRange.cols()); x++) {
#endif
            // Compute a ray for the current pixel.
            const glm::vec2 pixelPos = glm::vec2(x, y) / glm::vec2(m_config.renderResolution);
            Ray ray = m_pCamera->generateRay(pixelPos * 2.0f - 1.0f);

            // Compute where the ray enters and exists the volume.
            // If the ray misses the volume then we continue to the next pixel.
            if (!instersectRayVolumeBounds(ray, bounds))
                continue;

            // Get a color for the current pixel according to the current render mode.
            glm::vec4 color {};
            switch (m_config.renderMode) {
            case RenderMode::RenderSlicer: {
                color = traceRaySlice(ray, volumeCenter, planeNormal);
                break;
            }
            case RenderMode::RenderMIP: {
                color = traceRayMIP(ray, sampleStep);
                break;
            }
            case RenderMode::RenderComposite: {
                color = traceRayComposite(ray, sampleStep);
                break;
            }
            case RenderMode::RenderIso: {
                color = traceRayISO(ray, sampleStep);
                break;
            }
            case RenderMode::RenderTF2D: {
                color = traceRayTF2D(ray, sampleStep);
                break;
            }
            case RenderMode::RenderMIDA: {
                color = traceRayMIDA(ray, sampleStep);
                break;
            }
            case RenderMode::RenderCombined: {
                color = traceRayCombined(ray, sampleStep);
                break;
            };
            }
            // Write the resulting color to the screen.
            fillColor(x, y, color);

#if PARALLELISM == 1
        }
    }
});
#else
            }
        }
#endif
}

// ======= DO NOT MODIFY THIS FUNCTION ========
// This function generates a view alongside a plane perpendicular to the camera through the center of the volume
//  using the slicing technique.
glm::vec4 Renderer::traceRaySlice(const Ray& ray, const glm::vec3& volumeCenter, const glm::vec3& planeNormal) const
{
    const float t = glm::dot(volumeCenter - ray.origin, planeNormal) / glm::dot(ray.direction, planeNormal);
    const glm::vec3 samplePos = ray.origin + ray.direction * t;
    const float val = m_pVolume->getSampleInterpolate(samplePos);
    return glm::vec4(glm::vec3(std::max(val / m_pVolume->maximum(), 0.0f)), 1.f);
}

// ======= DO NOT MODIFY THIS FUNCTION ========
// Function that implements maximum-intensity-projection (MIP) raycasting.
// It returns the color assigned to a ray/pixel given it's origin, direction and the distances
// at which it enters/exits the volume (ray.tmin & ray.tmax respectively).
// The ray must be sampled with a distance defined by the sampleStep
glm::vec4 Renderer::traceRayMIP(const Ray& ray, float sampleStep) const
{
    float maxVal = 0.0f;

    // Incrementing samplePos directly instead of recomputing it each frame gives a measureable speed-up.
    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;
    const glm::vec3 increment = sampleStep * ray.direction;
    for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {
        const float val = m_pVolume->getSampleInterpolate(samplePos);
        maxVal = std::max(val, maxVal);
    }

    // Normalize the result to a range of [0 to mpVolume->maximum()].
    return glm::vec4(glm::vec3(maxVal) / m_pVolume->maximum(), 1.0f);
}

//EXTENSION 1: Maximum Intensity Difference Accumulation (MIDA)
glm::vec4 Renderer::traceRayMIDA(const Ray& ray, float sampleStep) const
{
    
    float maxVal = 0.0f;
    
    // The current position along the ray.
     glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;

    // The increment in the ray direction for each sample.
    const glm::vec3 increment = sampleStep * ray.direction;

    // The accumulated opacity along the ray.
    float accumulatedOpacity = 0.0f;

    // The accumulated color along the ray.
    glm::vec4 accumulatedColor(0.0f);

    for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {

        float val = m_pVolume->getSampleInterpolate(samplePos);
        float normalizedVal = val / m_pVolume->maximum();
        float normalizedMaxVal = maxVal / m_pVolume->maximum();

        const glm::vec4 tfValue = getTFValue(val);
        const glm::vec3 tfColor = glm::vec3(tfValue);
        const float tfOpacity = tfValue.a;

        glm::vec3 finalColor(0.0f);

        //EXTENSION 2: Volume shading + smoothstep
        if (m_config.volumeShading){ //if volume shading is enabled
      
            volume::GradientVoxel gradient = m_pGradientVolume->getGradientInterpolate(samplePos);
            glm::vec3 V = glm::normalize(m_pCamera->position() - samplePos); // View vector
            glm::vec3 L = glm::normalize(samplePos - ray.origin ); // Light vector

            glm::vec3 phongShading = computePhongShading(tfColor, gradient, L, V, m_config.ka, m_config.kd, m_config.ks, m_config.alpha);

            if(m_config.smoothstep){ //if smoothstep is enabled
                float gradientMagnitude = gradient.magnitude;
                float weight = glm::smoothstep(m_config.gl * m_pGradientVolume->maxMagnitude(), m_config.gh * m_pGradientVolume->maxMagnitude(), gradientMagnitude);
                finalColor = glm::mix(tfColor, phongShading, weight);
            }
            else{ //if smoothstep is disabled
                finalColor = phongShading;
            }
            
        }

        else{ //if volume shading is disabled
            finalColor = tfColor;
        }

        float delta = 0.0f;
        if (val > maxVal) {
            delta = normalizedVal - normalizedMaxVal;
            
        }

        float beta = 1.0f - delta;
        
        
        accumulatedColor = beta*accumulatedColor + (1 - beta*accumulatedOpacity) * tfOpacity * glm::vec4(finalColor, 1.0f);
        accumulatedOpacity = beta*accumulatedOpacity + (1 - beta*accumulatedOpacity) * tfOpacity;

        maxVal = std::max(val, maxVal);

    }

    
    return accumulatedColor;
    
}

//EXTENSION 2: MIDA TO DVR + MIDA TO MIP
glm::vec4 Renderer::traceRayCombined(const Ray& ray, float sampleStep) const
{
    float gamma = m_config.gamma;
    float maxVal = 0.0f;
    
    // The current position along the ray.
     glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;

    // The increment in the ray direction for each sample.
    const glm::vec3 increment = sampleStep * ray.direction;

    // The accumulated opacity along the ray.
    float accumulatedOpacity = 0.0f;

    // The accumulated color along the ray.
    glm::vec4 accumulatedColor(0.0f);

    for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {

        float val = m_pVolume->getSampleInterpolate(samplePos);
        float normalizedVal = val / m_pVolume->maximum();
        float normalizedMaxVal = maxVal / m_pVolume->maximum();

        const glm::vec4 tfValue = getTFValue(val);
        const glm::vec3 tfColor = glm::vec3(tfValue);
        const float tfOpacity = tfValue.a;

        glm::vec3 finalColor(0.0f);

        //EXTENSION 2: Volume shading + smoothstep
        if (m_config.volumeShading){ //if volume shading is enabled
      
            volume::GradientVoxel gradient = m_pGradientVolume->getGradientInterpolate(samplePos);
            glm::vec3 V = glm::normalize(m_pCamera->position() - samplePos); // View vector
            glm::vec3 L = glm::normalize(samplePos - ray.origin ); // Light vector

            glm::vec3 phongShading = computePhongShading(tfColor, gradient, L, V, m_config.ka, m_config.kd, m_config.ks, m_config.alpha);

            if(m_config.smoothstep){ //if smoothstep is enabled
                float gradientMagnitude = gradient.magnitude;
                float weight = glm::smoothstep(m_config.gl * m_pGradientVolume->maxMagnitude(), m_config.gh * m_pGradientVolume->maxMagnitude(), gradientMagnitude);
                finalColor = glm::mix(tfColor, phongShading, weight); //linear interpolation
            }
            else{ //if smoothstep is disabled
                finalColor = phongShading;
            }
            
        }

        else{ //if volume shading is disabled
            finalColor = tfColor;
        }

        float delta = 0.0f;
        if (val > maxVal) {
            delta = normalizedVal - normalizedMaxVal;
        }

        float beta;
        if (gamma <= 0.0f) { //MIDA TO DVR
            beta = 1.0f - delta*(1 + gamma);
        }
        else{ //MIDA TO MIP (see return statement)
            beta = 1.0f - delta;
        }
        
        accumulatedColor = beta*accumulatedColor + (1 - beta*accumulatedOpacity) * tfOpacity * glm::vec4(finalColor, 1.0f);
        accumulatedOpacity = beta*accumulatedOpacity + (1 - beta*accumulatedOpacity) * tfOpacity;

        maxVal = std::max(val, maxVal);

    }

    if (gamma <= 0.0f) { //MIDA TO DVR
        return accumulatedColor;
    }
    else { //MIDA TO MIP
        return glm::mix(accumulatedColor, glm::vec4(glm::vec3(maxVal) / m_pVolume->maximum(), 1.0f), gamma);
    }

}




// ======= TODO: IMPLEMENT ========
// This function should find the position where the ray intersects with the volume's isosurface.
// If volume shading is DISABLED then simply return the isoColor.
// If volume shading is ENABLED then return the phong-shaded color at that location using the local gradient (from m_pGradientVolume).
//   Use the camera position (m_pCamera->position()) as the light position.
// Use the bisectionAccuracy function (to be implemented) to get a more precise isosurface location between two steps.
glm::vec4 Renderer::traceRayISO(const Ray& ray, float sampleStep) const
{   
    const float R = 0.8f;
    const float G = 0.8f;
    const float B = 0.0f;

    auto color = glm::vec3(R, G, B);
 
    //if volume shading is disabled, then simply return the isoColor from the isoValue
    if (!m_config.volumeShading){
        
        // The current position along the ray.
        glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;

        // The increment in the ray direction for each sample.
        const glm::vec3 increment = sampleStep * ray.direction;

        float res = 0.0f;

        for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {
            
            // Get the volume value at the current sample position.
            float val = m_pVolume->getSampleInterpolate(samplePos);
            
            // If the value at the current sample position is greater than the iso value then we have found the isosurface.
            if (val > m_config.isoValue) {

                //unica cosa di cui non sono sicuro, nell'esempio la superficie è gialla mentre a me è bianca
                res = 1.0f;
                break;
                
            }
            
        }
        return glm::vec4(color * res, 1.0f);

    
    }

    //if volume shading is enabled, then return the phong-shaded color 
    //at that location using the local gradient (from m_pGradientVolume)
    else {

        glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;
        const glm::vec3 increment = sampleStep * ray.direction;

        for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {

            float val1 = m_pVolume->getSampleInterpolate(samplePos);
            float val2 = m_pVolume->getSampleInterpolate(samplePos + increment);

            // If the isosurface might be between the current and next sample positions
            if (val1 > m_config.isoValue || val2 > m_config.isoValue) {

                float preciseT = bisectionAccuracy(ray, t, t + sampleStep, m_config.isoValue);
                glm::vec3 precisePos = ray.origin + preciseT * ray.direction;

                volume::GradientVoxel gradient = m_pGradientVolume->getGradientInterpolate(precisePos);
                glm::vec3 V = glm::normalize(m_pCamera->position() - precisePos); // View vector
                glm::vec3 L = glm::normalize(precisePos - ray.origin ); // Light vector

                glm::vec3 phongShading = computePhongShading(color, gradient, L, V, m_config.ka, m_config.kd, m_config.ks, m_config.alpha);

                return glm::vec4(phongShading, 1.0f); 
            }

        }

        return glm::vec4(glm::vec3(0.0f), 1.0f); // Return default color if no intersection found
}

    
    

}

// Given that the iso value lies somewhere between t0 and t1, find a t for which the value
// closely matches the iso value (less than 0.01 difference). Add a limit to the number of
// iterations such that it does not get stuck in degerate cases.
float Renderer::bisectionAccuracy(const Ray& ray, float t0, float t1, float isoValue) const
{   
    static constexpr int maxIterations = 30; // Maximum number of iterations

    float precision = 0.01f; // Precision of the result
    
    float a = t0; // Start of the interval
    float b = t1; // End of the interval
    float c;      // Midpoint of the interval
    float fc;     // Function value at midpoint

    for (int iteration = 0; iteration < maxIterations; iteration++) {
        c = (a + b) / 2.0f; // Compute the midpoint of the interval

        // Compute the value at the midpoint
        fc = m_pVolume->getSampleInterpolate(ray.origin + c * ray.direction);

        // Check if the value at midpoint is close enough to isoValue or if the interval is sufficiently small
        if (std::abs(fc - isoValue) < precision || std::abs(b - a) < precision) {
            break; // Terminate if close to desired value or interval is too small
        }

        // Narrow the search interval
        if (fc < isoValue) {
            a = c; // Value lies in the upper half
        } else {
            b = c; // Value lies in the lower half
        }
    }

    return c; // Return the midpoint of the interval
}

// ======= TODO: IMPLEMENT ========
// Compute Phong Shading given the voxel color (material color), the gradient, the light vector and view vector.
// You can find out more about the Phong shading model at:
// https://en.wikipedia.org/wiki/Phong_reflection_model
//
// Use the given color for the ambient/specular/diffuse (you are allowed to scale these constants by a scalar value).
// You are free to choose any specular power that you'd like.
glm::vec3 Renderer::computePhongShading(const glm::vec3& color, const volume::GradientVoxel& gradient, const glm::vec3& L, const glm::vec3& V, float ka, float kd, float ks, float alpha)
{   
    // ambient 
    glm::vec3 ambient = ka * color;

    // diffuse
    float cos_theta = glm::dot(glm::normalize(gradient.dir), L);
    glm::vec3 diffuse = (kd * color * std::abs(cos_theta));
    
    // check if diffuse contains nan
    if (glm::any(glm::isnan(diffuse))) {
        diffuse = glm::vec3(0.0f);
    }

    // specular
    float cos_phi = glm::dot(glm::normalize(glm::reflect(L, gradient.dir)), V);
    glm::vec3 specular = ks * (glm::vec3(1.0)) * std::pow(std::abs(cos_phi), alpha);

    // return ambient;
    return  (ambient + diffuse + specular);
    
}

// ======= TODO: IMPLEMENT ========
// In this function, implement 1D transfer function raycasting.
// Use getTFValue to compute the color for a given volume value according to the 1D transfer function.

glm::vec4 Renderer::traceRayComposite(const Ray& ray, float sampleStep) const
{

    // The current position along the ray.
    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;

    // The increment in the ray direction for each sample.
    const glm::vec3 increment = sampleStep * ray.direction;

    // The accumulated opacity along the ray.
    float accumulatedOpacity = 0.0f;

    // The accumulated color along the ray.
    glm::vec4 accumulatedColor(0.0f);

    for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {
        // Get the volume value at the current sample position.
        const float val = m_pVolume->getSampleInterpolate(samplePos);

        // Get the color and opacity from the 1D transfer function.
        const glm::vec4 tfValue = getTFValue(val);
        glm::vec3 tfColor = glm::vec3(tfValue);
        const float tfOpacity = tfValue.a;

        if (m_config.volumeShading)
        {
            glm::vec3 precisePos = ray.origin + t * ray.direction;

            volume::GradientVoxel gradient = m_pGradientVolume->getGradientInterpolate(precisePos);
            glm::vec3 V = glm::normalize(m_pCamera->position() - precisePos); // View vector
            glm::vec3 L = glm::normalize(precisePos - ray.origin ); // Light vector

            tfColor = computePhongShading(tfColor, gradient, L, V, m_config.ka, m_config.kd, m_config.ks, m_config.alpha);
        }

        // Accumulate the color and opacity along the ray.
        accumulatedColor += (1.0f - accumulatedOpacity) * tfOpacity * glm::vec4(tfColor, 1.0f);
        accumulatedOpacity += (1.0f - accumulatedOpacity) * tfOpacity;

        // If the accumulated opacity is 1.0f then we can stop tracing the ray.
        if (accumulatedOpacity >= 1.0f)
            break;
    }

    // Return the accumulated color.
    return accumulatedColor;
}


// ======= DO NOT MODIFY THIS FUNCTION ========
// Looks up the color+opacity corresponding to the given volume value from the 1D tranfer function LUT (m_config.tfColorMap).
// The value will initially range from (m_config.tfColorMapIndexStart) to (m_config.tfColorMapIndexStart + m_config.tfColorMapIndexRange) .
glm::vec4 Renderer::getTFValue(float val) const
{
    // Map value from [m_config.tfColorMapIndexStart, m_config.tfColorMapIndexStart + m_config.tfColorMapIndexRange) to [0, 1) .
    const float range01 = (val - m_config.tfColorMapIndexStart) / m_config.tfColorMapIndexRange;
    const size_t i = std::min(static_cast<size_t>(range01 * static_cast<float>(m_config.tfColorMap.size())), m_config.tfColorMap.size() - 1);
    return m_config.tfColorMap[i];
}

// ======= TODO: IMPLEMENT ========
// In this function, implement 2D transfer function raycasting.
// Use the getTF2DOpacity function that you implemented to compute the opacity according to the 2D transfer function.

glm::vec4 Renderer::traceRayTF2D(const Ray& ray, float sampleStep) const
{
    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;
    const glm::vec3 increment = sampleStep * ray.direction;
    float accumulatedOpacity = 0.0f;

    for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {

        auto val = m_pVolume->getSampleInterpolate(samplePos);
        auto gradient = m_pGradientVolume->getGradientInterpolate(samplePos);
        auto magnitude = gradient.magnitude;

        const float tfOpacity = getTF2DOpacity(val, magnitude);
        
        accumulatedOpacity += (1.0f - accumulatedOpacity) * tfOpacity * m_config.TF2DColor.a;

        if (accumulatedOpacity >= 1.0f){
            accumulatedOpacity = 1.0f;
            break;
        }
    }

    return m_config.TF2DColor * accumulatedOpacity;
}


// ======= TODO: IMPLEMENT ========
// This function should return an opacity value for the given intensity and gradient according to the 2D transfer function.
// Calculate whether the values are within the radius/intensity triangle defined in the 2D transfer function widget.
// If so: return a tent weighting as described in the assignment
// Otherwise: return 0.0f

// The 2D transfer function settings can be accessed through m_config.TF2DIntensity and m_config.TF2DRadius.
float Renderer::getTF2DOpacity(float intensity, float gradientMagnitude) const
{   
    
    float apexIntensity = m_config.TF2DIntensity;
    float apexGradientMagnitude = m_pGradientVolume->minMagnitude();

    float baseIntensity1 = apexIntensity - m_config.TF2DRadius;
    float baseIntensity2 = apexIntensity + m_config.TF2DRadius;
    float baseGradientMagnitude = m_pGradientVolume->maxMagnitude();
    
    //calculate the line that connects the first base point to the apex
    float m1 = (apexGradientMagnitude - baseGradientMagnitude) / (apexIntensity - baseIntensity1);
    float q1 = apexGradientMagnitude - m1 * apexIntensity;

    //calculate the line that connects the second base point to the apex
    float m2 = (apexGradientMagnitude - baseGradientMagnitude) / (apexIntensity - baseIntensity2);
    float q2 = apexGradientMagnitude - m2 * apexIntensity;

    //calculate the projection 
    float projection;

    //check if the point is inside the triangle
    if (gradientMagnitude > m1 * intensity + q1 && gradientMagnitude > m2 * intensity + q2 && gradientMagnitude < baseGradientMagnitude && intensity > baseIntensity1 && intensity < baseIntensity2) {
        //TODO: return a tent weighting as follows:
        //set the values on the vertical line through the apex of the triangle to an opacity of 1 and from there towards the diagonal borders fall off to an opacity of zero by creating a linear transition that is aligned horizontally.
        if(intensity < apexIntensity){
            projection = (gradientMagnitude - q1) / m1;
        }
        else{
            projection = ((gradientMagnitude - q2) / m2);
        }

        float distancefromApex = std::abs(projection - apexIntensity);
        
        return (1.0f - std::abs(intensity - apexIntensity) / distancefromApex);
        
    }
    else {
        return 0.0f;
    }
    
    
}


// This function computes if a ray intersects with the axis-aligned bounding box around the volume.
// If the ray intersects then tmin/tmax are set to the distance at which the ray hits/exists the
// volume and true is returned. If the ray misses the volume the the function returns false.
//
// If you are interested you can learn about it at.
// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
bool Renderer::instersectRayVolumeBounds(Ray& ray, const Bounds& bounds) const
{
    const glm::vec3 invDir = 1.0f / ray.direction;
    const glm::bvec3 sign = glm::lessThan(invDir, glm::vec3(0.0f));

    float tmin = (bounds.lowerUpper[sign[0]].x - ray.origin.x) * invDir.x;
    float tmax = (bounds.lowerUpper[!sign[0]].x - ray.origin.x) * invDir.x;
    const float tymin = (bounds.lowerUpper[sign[1]].y - ray.origin.y) * invDir.y;
    const float tymax = (bounds.lowerUpper[!sign[1]].y - ray.origin.y) * invDir.y;

    if ((tmin > tymax) || (tymin > tmax))
        return false;
    tmin = std::max(tmin, tymin);
    tmax = std::min(tmax, tymax);

    const float tzmin = (bounds.lowerUpper[sign[2]].z - ray.origin.z) * invDir.z;
    const float tzmax = (bounds.lowerUpper[!sign[2]].z - ray.origin.z) * invDir.z;

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    ray.tmin = std::max(tmin, tzmin);
    ray.tmax = std::min(tmax, tzmax);
    return true;
}

// This function inserts a color into the framebuffer at position x,y
void Renderer::fillColor(int x, int y, const glm::vec4& color)
{
    const size_t index = static_cast<size_t>(m_config.renderResolution.x * y + x);
    m_frameBuffer[index] = color;
}
}