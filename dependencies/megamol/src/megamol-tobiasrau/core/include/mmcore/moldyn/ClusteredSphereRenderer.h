/*
 * ClusteredSphereRenderer.h
 *
 * Copyright (C) 2014 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/moldyn/AbstractSimpleSphereRenderer.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/Map.h"


namespace megamol {
namespace core {
namespace moldyn {

    /**
     * Renderer for simple sphere glyphs
     */
	class ClusteredSphereRenderer : public AbstractSimpleSphereRenderer {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "ClusteredSphereRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Renderer for clustered sphere glyphs.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
#ifdef _WIN32
#if defined(DEBUG) || defined(_DEBUG)
            HDC dc = ::wglGetCurrentDC();
            HGLRC rc = ::wglGetCurrentContext();
            ASSERT(dc != NULL);
            ASSERT(rc != NULL);
#endif // DEBUG || _DEBUG
#endif // _WIN32
            return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable();
        }

        /** Ctor. */
		ClusteredSphereRenderer(void);

        /** Dtor. */
		virtual ~ClusteredSphereRenderer(void);

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

        /**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool Render(Call& call);

    private:

        /** The sphere shader */
        vislib::graphics::gl::GLSLShader sphereShader;

    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

