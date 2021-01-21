extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{braced, parse_macro_input, token, Ident, Result};

mod kw {
    syn::custom_keyword!(SampledImage);
    syn::custom_keyword!(StorageImage);
    syn::custom_keyword!(UniformData);
    syn::custom_keyword!(StorageBuffer);
    syn::custom_keyword!(AccelerationStructure);
}

enum BindingType {
    SampledImage {
        _kw_token: kw::SampledImage,
    },
    StorageImage {
        _kw_token: kw::StorageImage,
    },
    UniformData {
        _kw_token: kw::UniformData,
        _lt_token: token::Lt,
        ty: Ident,
        _gt_token: token::Gt,
    },
    StorageBuffer {
        _kw_token: kw::StorageBuffer,
    },
    AccelerationStructure {
        _kw_token: kw::AccelerationStructure,
    },
}

struct Binding {
    name: Ident,
    _colon_token: token::Colon,
    ty: BindingType,
}

struct Layout {
    name: Ident,
    _brace_token: token::Brace,
    bindings: Punctuated<Binding, token::Comma>,
}

impl Parse for BindingType {
    fn parse(input: ParseStream) -> Result<Self> {
        let lookahead = input.lookahead1();
        if lookahead.peek(kw::SampledImage) {
            Ok(BindingType::SampledImage {
                _kw_token: input.parse()?,
            })
        } else if lookahead.peek(kw::StorageImage) {
            Ok(BindingType::StorageImage {
                _kw_token: input.parse()?,
            })
        } else if lookahead.peek(kw::UniformData) {
            Ok(BindingType::UniformData {
                _kw_token: input.parse()?,
                _lt_token: input.parse()?,
                ty: input.parse()?,
                _gt_token: input.parse()?,
            })
        } else if lookahead.peek(kw::StorageBuffer) {
            Ok(BindingType::StorageBuffer {
                _kw_token: input.parse()?,
            })
        } else if lookahead.peek(kw::AccelerationStructure) {
            Ok(BindingType::AccelerationStructure {
                _kw_token: input.parse()?,
            })
        } else {
            Err(lookahead.error())
        }
    }
}

impl Parse for Binding {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(Self {
            name: input.parse()?,
            _colon_token: input.parse()?,
            ty: input.parse()?,
        })
    }
}

impl Parse for Layout {
    fn parse(input: ParseStream) -> Result<Self> {
        let content;
        Ok(Self {
            name: input.parse()?,
            _brace_token: braced!(content in input),
            bindings: content.parse_terminated(Binding::parse)?,
        })
    }
}

impl Binding {
    fn get_binding(&self) -> (Option<TokenStream2>, TokenStream2) {
        match self.ty {
            BindingType::SampledImage { .. } => {
                let sampler = format_ident!("{}_sampler", self.name);
                (
                    Some(quote!(#sampler: vk::Sampler)),
                    quote!(DescriptorSetLayoutBinding::SampledImage { sampler: #sampler }),
                )
            }
            BindingType::StorageImage { .. } => (None, quote!(DescriptorSetLayoutBinding::StorageImage)),
            BindingType::UniformData { ref ty, .. } => (
                None,
                quote!(DescriptorSetLayoutBinding::UniformData {
                    size: ::std::mem::size_of::<#ty>() as u32,
                }),
            ),
            BindingType::StorageBuffer { .. } => (None, quote!(DescriptorSetLayoutBinding::StorageBuffer)),
            BindingType::AccelerationStructure { .. } => {
                (None, quote!(DescriptorSetLayoutBinding::AccelerationStructure))
            }
        }
    }

    fn get_data(&self) -> (TokenStream2, TokenStream2) {
        match self.ty {
            BindingType::SampledImage { .. } => {
                let image_view = format_ident!("{}_image_view", self.name);
                (
                    quote!(#image_view: vk::ImageView),
                    quote!(DescriptorSetBindingData::SampledImage { image_view: #image_view }),
                )
            }
            BindingType::StorageImage { .. } => {
                let image_view = format_ident!("{}_image_view", self.name);
                (
                    quote!(#image_view: vk::ImageView),
                    quote!(DescriptorSetBindingData::StorageImage { image_view: #image_view }),
                )
            }
            BindingType::UniformData { ref ty, .. } => {
                let writer = format_ident!("{}_writer", self.name);
                (
                    quote!(#writer: &dyn Fn(&mut #ty)),
                    quote!(DescriptorSetBindingData::UniformData {
                        size: ::std::mem::size_of::<#ty>() as u32,
                        writer: unsafe { ::std::mem::transmute(#writer) },
                    }),
                )
            }
            BindingType::StorageBuffer { .. } => {
                let buffer = format_ident!("{}_buffer", self.name);
                (
                    quote!(#buffer: vk::Buffer),
                    quote!(DescriptorSetBindingData::StorageBuffer { buffer: #buffer }),
                )
            }
            BindingType::AccelerationStructure { .. } => {
                let accel = format_ident!("{}_accel", self.name);
                (
                    quote!(#accel: vk::AccelerationStructureKHR),
                    quote!(DescriptorSetBindingData::AccelerationStructure { accel: #accel }),
                )
            }
        }
    }
}

#[proc_macro]
pub fn descriptor_set_layout(input: TokenStream) -> TokenStream {
    let Layout {
        name,
        _brace_token,
        bindings,
    } = parse_macro_input!(input as Layout);

    let (binding_args, binding_entries): (Vec<_>, Vec<_>) = bindings.iter().map(Binding::get_binding).unzip();
    let binding_args: Vec<_> = binding_args.iter().filter_map(|a| a.as_ref()).collect();
    let (data_args, data_entries): (Vec<_>, Vec<_>) = bindings.iter().map(Binding::get_data).unzip();

    quote!(
        struct #name(vk::DescriptorSetLayout);

        impl #name {
            pub fn new(descriptor_pool: &DescriptorPool, #(#binding_args),*) -> Self {
                let bindings = &[#(#binding_entries),*];
                Self(descriptor_pool.create_descriptor_set_layout(bindings))
            }

            pub fn write(&self, descriptor_pool: &DescriptorPool, #(#data_args),*) -> vk::DescriptorSet {
                let data = &[#(#data_entries),*];
                descriptor_pool.create_descriptor_set(self.0, data)
            }
        }
    )
    .into()
}
