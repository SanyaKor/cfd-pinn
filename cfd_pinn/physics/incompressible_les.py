from cfd_pinn.physics.base import PhysicsSystem
import torch

class IncompressibleLES(PhysicsSystem):


    def physics_residuals(self, coordinates: torch.Tensor, prediction: torch.Tensor, nut_values: torch.Tensor):

        uy = prediction[:, 0]
        uz = prediction[:, 1]
        p = prediction[:, 2]

        grad_uy = self.gradient(uy, coordinates)
        grad_uz = self.gradient(uz, coordinates)
        grad_p = self.gradient(p, coordinates)

        uy_y = grad_uy[:, 0]
        uy_z = grad_uy[:, 1]
        uy_t = grad_uy[:, 2]

        uz_y = grad_uz[:, 0]
        uz_z = grad_uz[:, 1]
        uz_t = grad_uz[:, 2]

        p_y = grad_p[:, 0]
        p_z = grad_p[:, 1]

        uy_yy = self.gradient(uy_y, coordinates)[:, 0]
        uy_zz = self.gradient(uy_z, coordinates)[:, 1]
        uz_yy = self.gradient(uz_y, coordinates)[:, 0]
        uz_zz = self.gradient(uz_z, coordinates)[:, 1]

        continuity = uy_y + uz_z

        #### NU = 1.5e-5  check out physical properties in constant folder
        nu_effective = 1.5e-5 + nut_values

        momentum_y = (
            uy_t + uy * uy_y + uz * uy_z + p_y
            - nu_effective * (uy_yy + uy_zz)
        )

        momentum_z = (
            uz_t + uy * uz_y + uz * uz_z + p_z
            - nu_effective * (uz_yy + uz_zz)
        )

        return continuity, momentum_y, momentum_z
