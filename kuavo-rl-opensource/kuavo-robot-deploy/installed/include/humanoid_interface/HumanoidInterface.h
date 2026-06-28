/******************************************************************************
Copyright (c) 2021, Farbod Farshidian. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

 * Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#pragma once

// ocs2
#include <ocs2_centroidal_model/FactoryFunctions.h>
#include <humanoid_interface/common/Types.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "humanoid_interface/common/ModelSettings.h"

/**
 * HumanoidInterface class
 * General interface for mpc implementation on the legged robot model
 */
namespace ocs2
{
  namespace humanoid
  {

    class HumanoidInterface final 
    {
    public:
      /**
       * Constructor
       *
       * @throw Invalid argument error if input task file or urdf file does not exist.
       *
       * @param [in] taskFile: The absolute path to the configuration file for the MPC.
       * @param [in] urdfFile: The absolute path to the URDF file for the robot.
       * @param [in] referenceFile: The absolute path to the reference configuration file.
       * @param [in] useHardFrictionConeConstraint: Which to use hard or soft friction cone constraints.
       */
      HumanoidInterface(const std::string &taskFile, const std::string &urdfFile, const std::string &referenceFile, const std::string &gaitFile,
                        int robot_version_int);

      ~HumanoidInterface() = default;

      const ModelSettings &modelSettings() const { return modelSettings_; }

      const vector_t &getInitialState() const { return initialState_; }
      PinocchioInterface &getPinocchioInterface() { return *pinocchioInterfacePtr_; }
      const CentroidalModelInfo &getCentroidalModelInfo() const { return centroidalModelInfo_; }

    private:

      ModelSettings modelSettings_;

      std::unique_ptr<PinocchioInterface> pinocchioInterfacePtr_;
      CentroidalModelInfo centroidalModelInfo_;

      vector_t initialState_;
      int robot_version_int_=34;//default to 34

    };

  } // namespace humanoid
} // namespace ocs2
