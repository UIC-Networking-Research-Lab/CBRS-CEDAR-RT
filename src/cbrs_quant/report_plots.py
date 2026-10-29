"""Vertical timing bars and annotated quality matrices for CBRS deployment comparisons."""
from __future__ import annotations

import math
from pathlib import Path

from .config import MODES

LABELS = {
    "torch_cuda_fp32": "PyTorch FP32", "torch_cuda_fp16": "PyTorch FP16",
    "torch_cpu_fp32": "PyTorch CPU", "ort_cuda_fp32": "ORT CUDA FP32",
    "ort_cuda_fp16": "ORT CUDA FP16", "ort_trt_fp16": "ORT TRT FP16",
    "ort_trt_int8": "ORT TRT INT8", "trt_fp32": "TensorRT FP32",
    "trt_fp16": "TensorRT FP16", "trt_int8": "TensorRT INT8",
    "ort_cpu_fp32": "ORT CPU FP32",
}
COLORS = {
    "torch_cuda_fp32": "#596579", "torch_cuda_fp16": "#9BA7B5",
    "torch_cpu_fp32": "#CFD5DC", "ort_cuda_fp32": "#235C95",
    "ort_cuda_fp16": "#64A2CD", "ort_trt_fp16": "#117B77",
    "ort_trt_int8": "#79BDB0", "trt_fp32": "#98714C",
    "trt_fp16": "#CB7435", "trt_int8": "#E5B35F", "ort_cpu_fp32": "#A7A193",
}
METRICS = (
    ("inference_time_per_frame_s", "Inference time per frame (s/frame)", "Lower is better"),
    ("macro_f1", "Detection F1", "Higher is better"),
    ("map50", "mAP@0.5", "Higher is better"),
)
SCOPE_LABELS = {
    "model_compute": "Model computation",
    "detector_pipeline": "Decoded image to detections",
    "image_to_detection": "PNG file to detections",
}


def model_label(name):
    if name.startswith("sig_yolo_"):
        return "CEDAR-YOLO-" + name.split("_")[2][-1]
    short = name.removeprefix("cbrs_yolo")
    return ("YOLOv" if short.startswith("8") else "YOLO") + short


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
        "text.color": "#172E43", "axes.labelcolor": "#44586A",
        "xtick.color": "#617385", "ytick.color": "#263D50",
        "axes.edgecolor": "#CAD3DC", "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    })
    return plt


def numeric(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def ordered_modes(rows):
    present = {r["mode"] for r in rows}
    return [m for m in MODES if m in present]


SHORT = {
    "torch_cuda_fp32": "Torch\nFP32", "torch_cuda_fp16": "Torch\nFP16",
    "torch_cpu_fp32": "Torch\nCPU", "ort_cuda_fp32": "CUDA\nFP32",
    "ort_cuda_fp16": "CUDA\nFP16", "ort_trt_fp16": "ORT TRT\nFP16",
    "ort_trt_int8": "ORT TRT\nINT8", "trt_fp32": "TRT\nFP32",
    "trt_fp16": "TRT\nFP16", "trt_int8": "TRT\nINT8", "ort_cpu_fp32": "ORT\nCPU",
}


def draw_time(ax, rows, modes):
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    lookup = {r["mode"]: r for r in rows}
    values = [numeric(lookup.get(m, {}).get("inference_time_per_frame_s")) for m in modes]
    maximum = max((v for v in values if v is not None), default=.01)
    for i, (mode, value) in enumerate(zip(modes, values)):
        if value is not None:
            ax.bar(i, value, width=.66, color=COLORS[mode], zorder=3,
                   edgecolor="white", linewidth=.55)
            ax.text(i, value+maximum*.025, f"{value:.5f}", ha="center", va="bottom",
                    rotation=90, fontsize=7.7, color="#263D50")
        else:
            ax.text(i, maximum*.025, "—", ha="center", fontsize=10, color="#81909D")
    ax.set_xticks(range(len(modes)), [SHORT[m] for m in modes])
    ax.tick_params(axis="x", length=0, pad=7, labelsize=7.5)
    ax.tick_params(axis="y", length=0, labelsize=8)
    ax.set_xlim(-.65,len(modes)-.35)
    ax.set_ylim(0,max(maximum*1.36,1e-8))
    ax.yaxis.set_major_locator(MaxNLocator(4, min_n_ticks=2))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f" if maximum>=.01 else "%.4f"))
    ax.yaxis.grid(True, color="#E1E7ED", linewidth=.65)
    ax.set_axisbelow(True)
    ax.set_ylabel("s/frame", fontsize=8.5)


def draw_quality(ax, matrix, columns, modes, title, *, fontsize=9.5):
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("cbrs_quality", ["#F1F4F6", "#B5D7D0", "#459C99", "#163F5A"])
    cmap.set_bad("#E8EBEF")
    values = np.asarray([[float("nan") if v is None else v for v in row] for row in matrix])
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(modes)), [LABELS[m] for m in modes])
    ax.tick_params(axis="both", length=0, pad=8, labelsize=9)
    for y in range(len(modes)):
        for x in range(len(columns)):
            v=values[y,x]
            ax.text(x,y,"—" if not np.isfinite(v) else f"{v:.4f}",ha="center",va="center",
                    fontsize=fontsize,color="white" if np.isfinite(v) and v>=.72 else "#284657")
    ax.set_xticks(np.arange(-.5,len(columns),1),minor=True)
    ax.set_yticks(np.arange(-.5,len(modes),1),minor=True)
    ax.grid(which="minor",color="white",linewidth=2.2)
    ax.tick_params(which="minor",bottom=False,left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title,loc="left",fontsize=12,weight="bold",pad=14)
    return im


def save(fig, stem: Path, formats, dpi):
    import matplotlib.pyplot as plt
    stem.parent.mkdir(parents=True,exist_ok=True)
    files=[]
    for extension in formats:
        path=stem.with_suffix("."+extension)
        fig.savefig(path,dpi=dpi,facecolor="white")
        files.append(path)
    plt.close(fig)
    return files


def model_chart(model, rows, output, scope, formats=("png","pdf"),dpi=180,suffix=""):
    plt=setup()
    batches=sorted({int(r["batch"]) for r in rows})
    if not batches:
        return []
    modes=ordered_modes(rows)
    fig=plt.figure(figsize=(14.8,9.2))
    outer=fig.add_gridspec(2,1,height_ratios=[1,1.12],hspace=.53,
                           left=.115,right=.95,bottom=.12,top=.86)
    top=outer[0].subgridspec(1,len(batches),wspace=.28)
    for i,batch in enumerate(batches):
        ax=fig.add_subplot(top[0,i])
        draw_time(ax,[r for r in rows if int(r["batch"])==batch],modes)
        ax.set_title(f"Batch {batch}",loc="left",fontsize=12,weight="bold",pad=14)
    bottom=outer[1].subgridspec(1,2,wspace=.42)
    lookup={(r["mode"],int(r["batch"])):r for r in rows}
    for i,(field,title) in enumerate((("macro_f1","Detection F1"),("map50","mAP@0.5"))):
        ax=fig.add_subplot(bottom[0,i])
        values=[[numeric(lookup.get((m,b),{}).get(field)) for b in batches] for m in modes]
        im=draw_quality(ax,values,[f"Batch {b}" for b in batches],modes,title,fontsize=10)
    cax=fig.add_axes([.955,.12,.012,.323])
    cb=fig.colorbar(im,cax=cax,ticks=[0,.5,1]);cb.outline.set_visible(False)
    cb.ax.tick_params(length=0,labelsize=8);cb.set_label("Quality · higher is better",fontsize=8)
    fig.suptitle(f"{model_label(model)}  |  {SCOPE_LABELS[scope]}",x=.035,y=.98,
                 ha="left",fontsize=18,weight="bold")
    fig.text(.035,.935,"Inference time per frame (s/frame)  ·  lower is better",fontsize=11,color="#617385")
    fig.text(.035,.035,"Time = 1 / mean throughput. F1 = mean of LTE and radar detection F1. "
             "Quality matrices use the full 0–1 scale; — = not measured.",fontsize=9,color="#617385")
    return save(fig,output/f"{model}_{scope}{suffix}",formats,dpi)


def overview(rows,models,output,scope="model_compute",batch=1,formats=("png","pdf"),dpi=180):
    plt=setup()
    selected=[r for r in rows if r["scope"]==scope and int(r["batch"])==batch]
    models=[m for m in models if any(r["model"]==m for r in selected)]
    if not models:
        return []
    modes=ordered_modes(selected);columns=min(3,len(models));count=math.ceil(len(models)/columns)
    fig,axes=plt.subplots(count,columns,figsize=(15.6,3.1*count+1),squeeze=False)
    for ax,model in zip(axes.flat,models):
        draw_time(ax,[r for r in selected if r["model"]==model],modes)
        ax.set_title(model_label(model),loc="left",fontsize=12,weight="bold",pad=12)
    for ax in list(axes.flat)[len(models):]:ax.set_visible(False)
    fig.suptitle("Inference time per frame (s/frame)",x=.035,y=.982,ha="left",fontsize=20,weight="bold")
    fig.text(.035,.951,f"{SCOPE_LABELS[scope]}  ·  batch {batch}  ·  lower is better",fontsize=11,color="#617385")
    fig.text(.035,.022,"Time = 1 / mean throughput. Each model has its own time scale. CUDA = ORT CUDA; TRT = direct TensorRT.",
             fontsize=9,color="#617385")
    fig.subplots_adjust(left=.075,right=.975,top=.90,bottom=.08,hspace=.52,wspace=.24)
    files=save(fig,output/f"overview_{scope}_b{batch:03d}_inference_time_per_frame_s",formats,dpi)
    lookup={(r["model"],r["mode"]):r for r in selected}
    for field,title in (("macro_f1","Detection F1"),("map50","mAP@0.5")):
        fig,ax=plt.subplots(figsize=(16,5.6))
        matrix=[[numeric(lookup.get((m,mode),{}).get(field)) for m in models] for mode in modes]
        im=draw_quality(ax,matrix,[model_label(m) for m in models],modes,"",fontsize=10)
        ax.tick_params(axis="x",labelrotation=35,labelsize=9)
        for tick in ax.get_xticklabels():tick.set_ha("right")
        fig.suptitle(title,x=.035,y=.965,ha="left",fontsize=21,weight="bold")
        fig.text(.035,.89,f"Batch {batch}  ·  rows: deployment policy  ·  columns: model  ·  higher is better",fontsize=11,color="#617385")
        cax=fig.add_axes([.958,.25,.012,.52]);cb=fig.colorbar(im,cax=cax,ticks=[0,.25,.5,.75,1])
        cb.outline.set_visible(False);cb.ax.tick_params(length=0,labelsize=9)
        note=("F1 averages LTE and radar detection F1 at confidence 0.25 and box IoU 0.5." if field=="macro_f1" else
              "mAP@0.5 averages the LTE and radar box AP at IoU 0.5.")
        fig.text(.035,.035,note+"  Exact values are printed in each cell; — = not measured.",fontsize=9,color="#617385")
        fig.subplots_adjust(left=.135,right=.945,top=.79,bottom=.25)
        # Fine separators group YOLOv8, YOLO11, YOLO26 and CEDAR without changing the scale.
        for x in (2.5,5.5,8.5):
            if x<len(models)-.5:ax.axvline(x,color="white",linewidth=5)
        files.extend(save(fig,output/f"overview_{scope}_b{batch:03d}_{field}",formats,dpi))
    return files


def coverage_chart(routes,output,formats=("png","pdf"),dpi=180):
    plt=setup();batches=sorted({int(r["batch"]) for r in routes})
    states=[s for s in ("passed","failed","pending","missing","invalid","incomplete") if any(r["status"]==s for r in routes)]
    states.extend(sorted({r["status"] for r in routes}-set(states)))
    palette={"passed":"#117B77","failed":"#C86746","pending":"#D5DFE7","missing":"#E5B35F"}
    fig,ax=plt.subplots(figsize=(10.5,4.7));bottom=[0]*len(batches)
    for state in states:
        counts=[sum(int(r["batch"])==b and r["status"]==state for r in routes) for b in batches]
        bars=ax.bar(range(len(batches)),counts,bottom=bottom,width=.62,label=state.replace("_"," "),color=palette.get(state,"#9BA7B5"))
        for bar,value in zip(bars,counts):
            if value:ax.text(bar.get_x()+bar.get_width()/2,bar.get_y()+bar.get_height()/2,str(value),ha="center",va="center",fontsize=10,
                             color="white" if state in ("passed","failed") else "#172E43")
        bottom=[a+b for a,b in zip(bottom,counts)]
    ax.set_xticks(range(len(batches)),[f"Batch {b}" for b in batches]);ax.set_ylabel("Planned model × policy combinations")
    ax.set_title("Experiment coverage",loc="left",fontsize=17,weight="bold",pad=18)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.12),ncol=len(states),frameon=False)
    fig.subplots_adjust(left=.10,right=.97,top=.86,bottom=.21)
    return save(fig,output/"coverage",formats,dpi)
