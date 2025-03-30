import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'Orbitron'


class LightDualityAnimation:
    def __init__(self):
        # Zmieniamy figurę: 2 wiersze, 1 kolumna
        self.fig = plt.figure(figsize=(12, 10), facecolor='black')
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

        # Oś (góra): Reprezentacja fali
        self.ax_wave = plt.subplot(gs[0, 0])
        self.ax_wave.set_title('Light as a Wave', fontsize=16, color='cyan', fontweight='bold')
        self.ax_wave.set_xlim(0, 10)
        self.ax_wave.set_ylim(-3, 3)
        self.ax_wave.set_xticks([])
        self.ax_wave.set_yticks([])
        self.ax_wave.axis('off')
        self.ax_wave.set_facecolor('black')

        # Oś (dół): Reprezentacja fotonów
        self.ax_photon = plt.subplot(gs[1, 0])
        self.ax_photon.set_title('Light as Particles (Photons)', fontsize=16, color='cyan', fontweight='bold')
        self.ax_photon.set_xlim(0, 10)
        self.ax_photon.set_ylim(-3, 3)
        self.ax_photon.set_xticks([])
        self.ax_photon.set_yticks([])
        self.ax_photon.axis('off')
        self.ax_photon.set_facecolor('black')

        # Pozycje źródła, ekranu i barier
        self.x_source = 0.5
        self.x_screen = 9.5
        self.x_slit = 5
        self.slit_width = 0.2
        self.slit_separation = 1.6

        # Usuwamy pionowe linie prowadzące (źródło i ekran)
        """
        for ax in [self.ax_wave, self.ax_photon]:
            ax.plot([self.x_source, self.x_source], [-3, 3], '--', color='#00FFAA', alpha=0.5, linewidth=1.5)
            ax.plot([self.x_screen, self.x_screen], [-3, 3], '--', color='#00FFAA', alpha=0.5, linewidth=1.5)
        """

        # Rysujemy barierę i dwie szczeliny
        for ax in [self.ax_wave, self.ax_photon]:
            ax.fill_between([self.x_slit, self.x_slit], -3, 3, color='#303060', alpha=0.9)
            slit1_center = self.slit_separation / 2
            slit2_center = -self.slit_separation / 2
            ax.fill_between([self.x_slit, self.x_slit],
                            slit1_center - self.slit_width / 2,
                            slit1_center + self.slit_width / 2,
                            color='#80FFFF')
            ax.fill_between([self.x_slit, self.x_slit],
                            slit2_center - self.slit_width / 2,
                            slit2_center + self.slit_width / 2,
                            color='#80FFFF')

        # Siatka do rysowania fali (wyższa rozdzielczość)
        self.grid_x = np.linspace(0, 10, 400)
        self.grid_y = np.linspace(-3, 3, 240)
        self.X, self.Y = np.meshgrid(self.grid_x, self.grid_y)
        self.Z = np.zeros_like(self.X)

        # Kolorowa mapa dla fali
        colors = [
            (0, 0.7, 1),   # jasny cyjan
            (0, 0.2, 0.6), # ciemniejszy niebieskawy
            (0.3, 0, 0.4), # odcień fioletu
            (1, 0.4, 0)    # pomarańcz
        ]
        self.wave_cmap = LinearSegmentedColormap.from_list('wave_colormap', colors, N=256)

        # pcolormesh do wizualizacji fali
        self.wave_plot = self.ax_wave.pcolormesh(
            self.X, self.Y, self.Z,
            cmap=self.wave_cmap,
            vmin=-1, vmax=1,
            shading='gouraud',
            rasterized=True,
            edgecolor='None'
        )

        # Scatter do fotonów
        self.photons = self.ax_photon.scatter([], [], s=50, color='#40F0FF', edgecolor='#00FFCC', zorder=5)
        self.photon_positions = []

        # EKRAN: intensywność (dopasowana do grid_y)
        self.screen_positions = np.linspace(-3, 3, 240)
        self.screen_intensity = np.zeros_like(self.screen_positions)
        self.intensity_plot, = self.ax_wave.plot([], [], alpha=0)

        # Fotonowe trafienia w ekran
        self.photon_hits = []
        self.screen_hits = self.ax_photon.scatter([], [], s=20, color='#00FFAA', alpha=0.7)

        # Histogram trafień fotonów
        self.hit_bins = np.linspace(-3, 3, 40)
        self.hit_counts = np.zeros(len(self.hit_bins) - 1)
        self.hist_plot = self.ax_photon.barh(
            (self.hit_bins[:-1] + self.hit_bins[1:]) / 2,
            self.hit_counts,
            height=self.hit_bins[1] - self.hit_bins[0],
            left=self.x_screen,
            color='#00FFAA',
            alpha=0.7
        )

        # Parametry fali
        self.t = 0
        self.wave_speed = 1.0
        self.wavelength = 0.5
        self.frequency = self.wave_speed / self.wavelength
        self.angular_frequency = 2 * np.pi * self.frequency

        # Adnotacja/tekst na dole (opcjonalnie)
        self.annotation = self.fig.text(0.5, 0.01, "", ha='center', fontsize=12, color='#00FFCC')

        plt.tight_layout()
        # Można dodatkowo dostosować marginesy w razie potrzeby:
        # self.fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)

    def calculate_wave_field(self, t):
        """Oblicza pole fali, wliczając interferencję z dwóch szczelin."""
        Z = np.zeros_like(self.X)

        # Pozycje szczelin
        slit1_y = self.slit_separation / 2
        slit2_y = -self.slit_separation / 2
        slit1_pos = (self.x_slit, slit1_y)
        slit2_pos = (self.x_slit, slit2_y)

        # Stałe fali
        k = 2 * np.pi / self.wavelength
        omega = self.angular_frequency

        # Część przed barierą
        mask_before = self.X < self.x_slit
        Z[mask_before] = np.sin(k * self.X[mask_before] - omega * t)
        edge_attenuation = np.exp(-((self.Y[mask_before]) ** 2) / 4)
        Z[mask_before] *= edge_attenuation

        # Część za szczelinami
        mask_after = self.X >= self.x_slit
        r1 = np.sqrt((self.X[mask_after] - slit1_pos[0]) ** 2 + (self.Y[mask_after] - slit1_pos[1]) ** 2)
        r2 = np.sqrt((self.X[mask_after] - slit2_pos[0]) ** 2 + (self.Y[mask_after] - slit2_pos[1]) ** 2)

        wave1 = np.sin(k * r1 - omega * (t - self.x_slit / self.wave_speed)) / np.sqrt(np.maximum(r1, 0.1))
        wave2 = np.sin(k * r2 - omega * (t - self.x_slit / self.wave_speed)) / np.sqrt(np.maximum(r2, 0.1))

        # Zero w barierze poza szczelinami
        y_dist1 = np.abs(self.Y[mask_after] - slit1_pos[1])
        y_dist2 = np.abs(self.Y[mask_after] - slit2_pos[1])
        at_slit1 = y_dist1 <= self.slit_width / 2
        at_slit2 = y_dist2 <= self.slit_width / 2
        at_barrier = np.abs(self.X[mask_after] - self.x_slit) < 0.1
        barrier_mask = at_barrier & ~(at_slit1 | at_slit2)
        wave1[barrier_mask] = 0
        wave2[barrier_mask] = 0

        Z[mask_after] = wave1 + wave2
        return Z

    def update_screen_pattern(self, Z):
        """Aktualizuje intensywność na ekranie (suma kwadratów amplitudy)."""
        screen_x_index = np.argmin(np.abs(self.grid_x - self.x_screen))
        screen_values = Z[:, screen_x_index]
        intensity = screen_values ** 2

        # Dodajemy część aktualnego stanu do pamięci intensywności
        self.screen_intensity = 0.95 * self.screen_intensity + 0.05 * intensity
        self.intensity_plot.set_data([], [])

    def generate_photon(self):
        """Losowe tworzenie fotonu przy źródle."""
        if np.random.random() < 0.3:
            self.photon_positions.append([self.x_source, 0, 0])  # x, y, 'który slit'

    def update_photons(self):
        """Poruszanie fotonów i rejestrowanie trafień w ekran."""
        new_positions = []
        screen_hits_x = []
        screen_hits_y = []

        for photon in self.photon_positions:
            x, y, slit_choice = photon

            # 1) Przelot do bariery
            if x < self.x_slit - 0.1:
                x += 0.2
                new_positions.append([x, y, slit_choice])

            # 2) Przejście przez szczeliny
            elif x < self.x_slit + 0.1:
                x += 0.2
                if slit_choice == 0:
                    slit_choice = 1 if np.random.random() < 0.5 else 2
                    if slit_choice == 1:
                        y = self.slit_separation / 2 + np.random.normal(0, self.slit_width / 4)
                    else:
                        y = -self.slit_separation / 2 + np.random.normal(0, self.slit_width / 4)
                new_positions.append([x, y, slit_choice])

            # 3) Od szczelin do ekranu
            elif x < self.x_screen:
                x += 0.2
                x_idx = int((x - 0) / 10 * (len(self.grid_x) - 1))
                y_idx = int((y + 3) / 6 * (len(self.grid_y) - 1))
                if 0 <= x_idx < self.Z.shape[1] and 0 <= y_idx < self.Z.shape[0]:
                    local_intensity = self.Z[y_idx, x_idx] ** 2
                    bias = np.sin(2 * np.pi * x / self.wavelength) * 0.05
                    # Zmieniamy y fotonu losowo, uwzględniając "bias"
                    y += np.random.normal(bias, 0.05 - 0.03 * min(local_intensity, 1))
                else:
                    y += np.random.normal(0, 0.05)
                new_positions.append([x, y, slit_choice])

            # 4) Dotarcie do ekranu
            else:
                screen_hits_x.append(x)
                screen_hits_y.append(y)
                self.photon_hits.append(y)
                bin_indices = np.digitize(y, self.hit_bins) - 1
                if 0 <= bin_indices < len(self.hit_counts):
                    self.hit_counts[bin_indices] += 1

        self.photon_positions = new_positions
        x_vals = [p[0] for p in new_positions]
        y_vals = [p[1] for p in new_positions]

        # Aktualizacja scattera trafień w ekran
        if screen_hits_x:
            old_offsets = self.screen_hits.get_offsets()
            if len(old_offsets) > 0:
                current_hits_x, current_hits_y = old_offsets.T
            else:
                current_hits_x, current_hits_y = [], []
            current_hits_x = list(current_hits_x) + screen_hits_x
            current_hits_y = list(current_hits_y) + screen_hits_y
            self.screen_hits.set_offsets(np.column_stack([current_hits_x, current_hits_y]))

        # Aktualizacja histogramu
        for rect, height in zip(self.hist_plot, self.hit_counts):
            rect.set_width(height * 0.05)

        return x_vals, y_vals

    def update(self, frame):
        """Funkcja wywoływana w każdej klatce animacji."""
        self.t += 0.1

        # Obliczamy i wizualizujemy falę
        self.Z = self.calculate_wave_field(self.t)
        self.wave_plot.set_array(self.Z.ravel())

        # Aktualizujemy wzorzec na ekranie (intensywność)
        self.update_screen_pattern(self.Z)

        # Generujemy i przemieszczamy fotony
        self.generate_photon()
        x_vals, y_vals = self.update_photons()
        if x_vals:
            self.photons.set_offsets(np.column_stack([x_vals, y_vals]))
        else:
            self.photons.set_offsets(np.empty((0, 2)))

        # Tekst/komentarz w czasie animacji (opcjonalnie)
        if frame < 20:
            self.annotation.set_text("")
        elif frame < 40:
            self.annotation.set_text("")
        elif frame < 60:
            self.annotation.set_text("")
        elif frame < 80:
            self.annotation.set_text("")
        elif frame < 100:
            self.annotation.set_text("")
        else:
            self.annotation.set_text("")

        # Elementy do zaktualizowania w blit
        return [self.wave_plot, self.photons, self.intensity_plot, self.screen_hits]

    def animate(self):
        """Uruchamia animację."""
        anim = FuncAnimation(
            self.fig,
            self.update,
            frames=120,
            interval=50,
            blit=True
        )
        plt.tight_layout()
        return anim


def main():
    # Opcjonalnie styl ciemny
    plt.style.use('dark_background')

    duality_animation = LightDualityAnimation()
    anim = duality_animation.animate()

    # Jeśli chcesz zapisać do pliku wideo, odkomentuj poniższą linię:
   # anim.save(r'C:\Users\topgu\PycharmProjects\obrazowanie\media\videos\light_duality.mp4', writer='ffmpeg', fps=20, dpi=150)

    plt.show()


if __name__ == "__main__":
    main()
