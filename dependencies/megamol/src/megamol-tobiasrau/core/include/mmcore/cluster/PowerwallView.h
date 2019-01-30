/*
 * PowerwallView.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_POWERWALLVIEW_H_INCLUDED
#define MEGAMOLCORE_POWERWALLVIEW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/cluster/AbstractClusterView.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/graphics/gl/FramebufferObject.h"


namespace megamol {
namespace core {
namespace cluster {


    /**
     * Abstract base class of override rendering views
     */
    class PowerwallView : public AbstractClusterView {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "PowerwallView";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Powerwall View Module";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /**
         * Disallow usage in quickstarts
         *
         * @return false
         */
        static bool SupportQuickstart(void) {
            return false;
        }

        /** Ctor. */
        PowerwallView(void);

        /** Dtor. */
        virtual ~PowerwallView(void);

        /**
         * Renders this AbstractView3D in the currently active OpenGL context.
         *
         * @param context
         */
        virtual void Render(const mmcRenderViewContext& context);

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
         * A message has been received over the control channel.
         *
         * @param sender The sending object
         * @param msg The received message
         */
        virtual void OnCommChannelMessage(CommChannel& sender,
            const vislib::net::AbstractSimpleMessage& msg);

        /**
         * Gets the info message and icon for the fallback view
         *
         * @param outMsg The message to be shows in the fallback view
         * @param outState The state icon to be shows in the fallback view
         */
        virtual void getFallbackMessageInfo(vislib::TString& outMsg,
            InfoIconRenderer::IconState& outState);

    private:

        /** The pause flag for the view */
        bool pauseView;

        /** The fbo shown if the remote rendering is paused */
        vislib::graphics::gl::FramebufferObject *pauseFbo;

    };


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_POWERWALLVIEW_H_INCLUDED */
