import itertools

import numpy as np
import scipy.spatial
import scipy.cluster
import matplotlib.cm
import plotly.graph_objs as go
from sklearn.metrics import silhouette_score


class DendrogramCut:
    def __init__(self, k_max, method='average'):
        self.k_max = k_max
        self.method = method
            
    def fit(self, distance_matrix):
        '''
        Build linkage_stats
            css: cross sum of square when merging c1 and c2
            tss: total sum of square of merged cluster
        '''
        self.distance_matrix = distance_matrix
        self.n_data = distance_matrix.shape[0]
        self.linkage = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(distance_matrix), method=self.method, optimal_ordering=True)
        self.linkage_stats = [{'c1': 0, 'c2': 0, 'css': 0, 'tss': 0, 'indices': set()} for _ in range(2 * self.n_data - 1)]

        for i in range(self.n_data):
            self.linkage_stats[i]['c1'] = i
            self.linkage_stats[i]['c2'] = i
            self.linkage_stats[i]['indices'].add(i)

        for i, (c1, c2, _, _) in enumerate(self.linkage):
            c1 = int(c1)
            c2 = int(c2)
            self.linkage_stats[i + self.n_data]['c1'] = c1
            self.linkage_stats[i + self.n_data]['c2'] = c2
            self.linkage_stats[i + self.n_data]['indices'].update(self.linkage_stats[c1]['indices'])
            self.linkage_stats[i + self.n_data]['indices'].update(self.linkage_stats[c2]['indices'])

        for i in range(self.n_data, 2 * self.n_data - 1):
            c1 = self.linkage_stats[i]['c1']
            c2 = self.linkage_stats[i]['c2']
            c1_indices = np.asarray(list(self.linkage_stats[c1]['indices']))
            c2_indices = np.asarray(list(self.linkage_stats[c2]['indices']))

            sample_distances = distance_matrix[c1_indices, :][:, c2_indices]
            self.linkage_stats[i]['css'] = np.sum(sample_distances ** 2)
            self.linkage_stats[i]['tss'] = self.linkage_stats[i]['css'] + self.linkage_stats[c1]['tss'] + self.linkage_stats[c2]['tss']

        ### dynamic programming ###
        '''
        Dynamic programming
            kl_mat[i, k]: the number of clusters in the left branch of linkage_stat[$i], maximal cluster $k
            mss_mat[i, k]: the optimal mean square error achieved at linkage_stat[$i], maximal cluster $k
        '''
        self.kl_mat = np.zeros((self.n_data * 2 - 1, self.k_max + 1), dtype=int)
        self.mss_mat = np.zeros((self.n_data * 2 - 1, self.k_max + 1), dtype=float) + np.inf

        for i in range(self.n_data):
            self.mss_mat[i, 1] = 0

        for i in range(self.n_data, 2 * self.n_data - 1):
            self.mss_mat[i, 1] = self.linkage_stats[i]['tss'] / len(self.linkage_stats[i]['indices'])

        for i in range(self.n_data, 2 * self.n_data - 1):
            for k in range(2, self.k_max + 1):
                kl_min = 0
                mss_min = np.inf
                for kl in range(1, k):
                    tss = self.mss_mat[self.linkage_stats[i]['c1'], kl] + self.mss_mat[self.linkage_stats[i]['c2'], k - kl]
                    if tss < mss_min:
                        kl_min = kl
                        mss_min = tss
                self.kl_mat[i, k] = kl_min
                self.mss_mat[i, k] = mss_min
                
        return self

    def _get_cut_nodes(self, v, k):
        if k == 1:
            yield v
        else:
            yield from self._get_cut_nodes(self.linkage_stats[v]['c1'], self.kl_mat[v, k])
            yield from self._get_cut_nodes(self.linkage_stats[v]['c2'], k - self.kl_mat[v, k])

    def get_cluster_mss(self, k):
        total_mss = 0.
        for cid in self._get_cut_nodes(2 * self.n_data - 2, k):
            total_mss += self.linkage_stats[cid]['tss'] / len(self.linkage_stats[cid]['indices'])
        
        return total_mss

    def get_cluster_label(self, k):
        ### get flat clusters ###
        z = np.zeros(self.n_data, dtype=int) - 1
        for c, cid in enumerate(self._get_cut_nodes(2 * self.n_data - 2, k)):
            for i in self.linkage_stats[cid]['indices']:
                if z[i] != -1:
                    print(i)
                z[i] = c

        return z

    def _dirichlet_process_kl(self, n_list, alpha_):
        out = 0.
        count = alpha_
        for n in n_list:
            out -= np.log(alpha_ / count)
            count += 1
            for i in range(1, n):
                out -= np.log(i / count)
                count += 1
        return out

    def pac_bayesian_cut(self, alpha_=1., lambda_=1.):
        min_loss = np.inf
        min_loss_k = None

        for k in range(1, self.k_max + 1):
            n_list = [len(self.linkage_stats[c]['indices']) for c in self._get_cut_nodes(2 * self.n_data - 2, k)]
            total_mss = self.get_cluster_mss(k)
            loss = total_mss + self._dirichlet_process_kl(n_list, alpha_) / lambda_
            if loss < min_loss:
                min_loss = loss
                min_loss_k = k

        return min_loss_k
    
    def dendrogram_plot(self, k=None, label=None):
        cmap = matplotlib.cm.get_cmap('viridis')

        order = scipy.cluster.hierarchy.leaves_list(self.linkage)

        dendrogram_stat = dict((v, {'x': i, 'y': 0}) for i, v in enumerate(order))

        data = []

        ### plot dendrogram ###
        if k is None:
            norm = np.max(self.linkage[:, 2])

            for res in self.linkage:
                new_cluster = len(dendrogram_stat)
                dendrogram_stat[new_cluster] = {'x': (dendrogram_stat[res[0]]['x'] + dendrogram_stat[res[1]]['x']) / 2, 'y': res[2]}
                x_range = np.array([dendrogram_stat[res[0]]['x'], dendrogram_stat[res[0]]['x'], dendrogram_stat[res[1]]['x'], dendrogram_stat[res[1]]['x']])
                y_range = np.array([dendrogram_stat[res[0]]['y'], dendrogram_stat[new_cluster]['y'], dendrogram_stat[new_cluster]['y'], dendrogram_stat[res[1]]['y']])

                data.append(
                    {
                        'hoverinfo': 'text',
                        'marker': {'color': 'rgb({}, {}, {})'.format(*cmap(res[2] / norm, bytes=True)[:3])},
                        'mode': 'lines',
                        'type': 'scatter',
                        'x': x_range,
                        'xaxis': 'x',
                        'y': y_range,
                        'yaxis': 'y'
                    }
                )
        else:
            pivot = set(self._get_cut_nodes(2 * self.n_data - 2, k))
            z = self.get_cluster_label(k)
            colorscale = dict((i, z[i] / (k - 1)) for i in range(self.n_data))

            for res in self.linkage:
                new_cluster = len(dendrogram_stat)
                colorscale[new_cluster] = (colorscale[int(res[0])] + colorscale[int(res[1])]) / 2
                dendrogram_stat[new_cluster] = {'x': (dendrogram_stat[res[0]]['x'] + dendrogram_stat[res[1]]['x']) / 2, 'y': res[2]}
                x_range = np.array([dendrogram_stat[res[0]]['x'], dendrogram_stat[res[0]]['x'], dendrogram_stat[res[1]]['x'], dendrogram_stat[res[1]]['x']])
                y_range = np.array([dendrogram_stat[res[0]]['y'], dendrogram_stat[new_cluster]['y'], dendrogram_stat[new_cluster]['y'], dendrogram_stat[res[1]]['y']])

                if new_cluster in pivot:
                    data.append(
                        {
                            'x': [dendrogram_stat[new_cluster]['x']],
                            'y': [dendrogram_stat[new_cluster]['y']],
                            'type': 'scatter',
                            'mode': 'markers',
                            'xaxis': 'x',
                            'yaxis': 'y',
                            'marker': {'size': [20], 'color': 'rgb({}, {}, {})'.format(*cmap(colorscale[new_cluster], bytes=True)[:3])}
                        }
                    )

                data.append(
                    {
                        'hoverinfo': 'text',
                        'marker': {'color': 'rgb({}, {}, {})'.format(*cmap(colorscale[new_cluster], bytes=True)[:3])},
                        'mode': 'lines',
                        'type': 'scatter',
                        'x': x_range,
                        'xaxis': 'x',
                        'y': y_range,
                        'yaxis': 'y'
                    }
                )

        fig = go.Figure(data=data)

        ### edit layout ###
        if label is not None:
            fig.update_layout(
                {
                    'xaxis': {
                        'tickmode': 'array',
                        'tickvals': np.arange(len(label)),
                        'ticktext': label[order]
                    }
                }
            )
            
        fig.update_layout(
            {
                'width': 800,
                'height': 800,
                'showlegend': False,
                'hovermode': 'closest',
                'xaxis': {
                    'domain': [0, 1],
                    'rangemode': 'tozero',
                    'mirror': False,
                    'showgrid': False,
                    'showline': False,
                    'zeroline': False,
                    'showticklabels': True,
                    'ticks': ""
                },
                'yaxis': {
                    'domain': [0, 1],
                    'rangemode': 'tozero',
                    'mirror': False,
                    'showgrid': False,
                    'showline': False,
                    'zeroline': False,
                    'showticklabels': True,
                    'ticks': ""
                },
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(0,0,0,0)'
            }
        )

        fig.show()

    def heatmap_with_dendrogram_plot(self, k=None, label=None):
        cmap = matplotlib.cm.get_cmap('viridis')
        
        order = scipy.cluster.hierarchy.leaves_list(self.linkage)

        dendrogram_stat = dict((v, {'x': i, 'y': 0}) for i, v in enumerate(order))

        data = []

        ### plot dendrogram ###
        if k is None:
            norm = np.max(self.linkage[:, 2])

            for res in self.linkage:
                new_cluster = len(dendrogram_stat)
                dendrogram_stat[new_cluster] = {'x': (dendrogram_stat[res[0]]['x'] + dendrogram_stat[res[1]]['x']) / 2, 'y': res[2]}
                x_range = np.array([dendrogram_stat[res[0]]['x'], dendrogram_stat[res[0]]['x'], dendrogram_stat[res[1]]['x'], dendrogram_stat[res[1]]['x']])
                y_range = np.array([dendrogram_stat[res[0]]['y'], dendrogram_stat[new_cluster]['y'], dendrogram_stat[new_cluster]['y'], dendrogram_stat[res[1]]['y']])

                data.append(
                    {
                        'hoverinfo': 'text',
                        'marker': {'color': 'rgb({}, {}, {})'.format(*cmap(res[2] / norm, bytes=True)[:3])},
                        'mode': 'lines',
                        'type': 'scatter',
                        'x': x_range,
                        'xaxis': 'x',
                        'y': y_range,
                        'yaxis': 'y2'
                    }
                )

                data.append(
                    {
                        'hoverinfo': 'text',
                        'marker': {'color': 'rgb({}, {}, {})'.format(*cmap(res[2] / norm, bytes=True)[:3])},
                        'mode': 'lines',
                        'type': 'scatter',
                        'x': -y_range,
                        'xaxis': 'x2',
                        'y': x_range,
                        'yaxis': 'y'
                    }
                )
        else:
            pivot = set(self._get_cut_nodes(2 * self.n_data - 2, k))
            z = self.get_cluster_label(k)
            colorscale = dict((i, z[i] / (k - 1)) for i in range(self.n_data))

            for res in self.linkage:
                new_cluster = len(dendrogram_stat)
                colorscale[new_cluster] = (colorscale[int(res[0])] + colorscale[int(res[1])]) / 2
                dendrogram_stat[new_cluster] = {'x': (dendrogram_stat[res[0]]['x'] + dendrogram_stat[res[1]]['x']) / 2, 'y': res[2]}
                x_range = np.array([dendrogram_stat[res[0]]['x'], dendrogram_stat[res[0]]['x'], dendrogram_stat[res[1]]['x'], dendrogram_stat[res[1]]['x']])
                y_range = np.array([dendrogram_stat[res[0]]['y'], dendrogram_stat[new_cluster]['y'], dendrogram_stat[new_cluster]['y'], dendrogram_stat[res[1]]['y']])

                if new_cluster in pivot:
                    data.append(
                        {
                            'x': [dendrogram_stat[new_cluster]['x']],
                            'y': [dendrogram_stat[new_cluster]['y']],
                            'type': 'scatter',
                            'mode': 'markers',
                            'xaxis': 'x',
                            'yaxis': 'y2',
                            'marker': {'size': [10], 'color': 'rgb({}, {}, {})'.format(*cmap(colorscale[new_cluster], bytes=True)[:3])}
                        }
                    )
                    data.append(
                        {
                            'x': [-dendrogram_stat[new_cluster]['y']],
                            'y': [dendrogram_stat[new_cluster]['x']],
                            'type': 'scatter',
                            'mode': 'markers',
                            'xaxis': 'x2',
                            'yaxis': 'y',
                            'marker': {'size': [10], 'color': 'rgb({}, {}, {})'.format(*cmap(colorscale[new_cluster], bytes=True)[:3])}
                        }
                    )

                data.append(
                    {
                        'hoverinfo': 'text',
                        'marker': {'color': 'rgb({}, {}, {})'.format(*cmap(colorscale[new_cluster], bytes=True)[:3])},
                        'mode': 'lines',
                        'type': 'scatter',
                        'x': x_range,
                        'xaxis': 'x',
                        'y': y_range,
                        'yaxis': 'y2'
                    }
                )

                data.append(
                    {
                        'hoverinfo': 'text',
                        'marker': {'color': 'rgb({}, {}, {})'.format(*cmap(colorscale[new_cluster], bytes=True)[:3])},
                        'mode': 'lines',
                        'type': 'scatter',
                        'x': -y_range,
                        'xaxis': 'x2',
                        'y': x_range,
                        'yaxis': 'y'
                    }
                )
                
            ### plot hard cluster region ###

            ordered_z = z[order]
            block_size = [sum(1 for __ in g) for _, g in itertools.groupby(ordered_z)]
            start = -0.5
            for i in range(len(block_size)):
                data.append(
                    go.Scatter(
                        x=[start, start, start+block_size[i], start+block_size[i], start],
                        y=[start, start+block_size[i], start+block_size[i], start, start],
                        mode='lines',
                        hoverinfo='none',
                        line=dict(color='#FF0000')
                    )
                )
                start += block_size[i]

        ### plot heatmap ###
        data.append(
            go.Heatmap(
                z=self.distance_matrix[order, :][:, order],
                colorscale='Blues',
                reversescale=True,
                hoverinfo='x+y+z'
            )
        )

        fig = go.Figure(data=data)

        ### edit layout ###
        if label is not None:
            fig.update_layout(
                {
                    'xaxis': {
                        'tickmode': 'array',
                        'tickvals': np.arange(len(label)),
                        'ticktext': label[order]
                    },
                    'yaxis': {
                        'tickmode': 'array',
                        'tickvals': np.arange(len(label)),
                        'ticktext': label[order],
                        'autorange': 'reversed'
                    }
                }
            )
            
        fig.update_layout(
            {
                'width': 800,
                'height': 800,
                'showlegend': False,
                'hovermode': 'closest',
                'xaxis': {
                    'domain': [.15, 1],
                    'rangemode': 'tozero',
                    'mirror': False,
                    'showgrid': False,
                    'showline': False,
                    'zeroline': False,
                    'showticklabels': False,
                    'ticks': ""
                },
                'xaxis2': {
                    'domain': [0, .15],
                    'rangemode': 'tozero',
                    'mirror': False,
                    'showgrid': False,
                    'showline': False,
                    'zeroline': False,
                    'showticklabels': False,
                    'ticks': ""
                },
                'yaxis': {
                    'domain': [0, .85],
                    'rangemode': 'tozero',
                    'mirror': False,
                    'showgrid': False,
                    'showline': False,
                    'zeroline': False,
                    'showticklabels': False,
                    'ticks': ""
                },
                'yaxis2': {
                    'domain':[.825, .975],
                    'rangemode': 'tozero',
                    'mirror': False,
                    'showgrid': False,
                    'showline': False,
                    'zeroline': False,
                    'showticklabels': False,
                    'ticks': ""
                }
            }
        )

        fig.show()
        
    def elbow_plot(self):
        x = np.arange(self.k_max) + 1

        y = np.asarray([sum(self.linkage_stats[i]['tss'] / len(self.linkage_stats[i]['indices']) for i in self._get_cut_nodes(-1, k)) for k in range(1, self.k_max + 1)])
        y = (y[0] - y) / y[0]
        
        fig = go.Figure(data=[go.Scatter(x=x, y=y, mode='lines')])
        fig.show()
        
    def silhouette_score_plot(self):
        x = np.arange(2, self.k_max + 1)
        y = [silhouette_score(self.distance_matrix, self.get_cluster_label(k), metric='precomputed') for k in x]
        
        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines'))
        fig.show()